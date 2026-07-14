# Start from the last coding stage of the previous LLM evals project
import json
import logging
import os
from functools import lru_cache

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.rails.llm.options import GenerationOptions

from cloud_config import load_app_config
from cloud_services import (
    create_redis_chat_history,
    initialize_langfuse_client,
    initialize_qdrant_store,
)

app = FastAPI(title="Smartphone Assistant API")
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str

    @field_validator("user_input", "user_id", "session_id")
    @classmethod
    def validate_non_empty_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


# Load environment variables from .env file
dotenv.load_dotenv()

# Use the per-user LiteLLM proxy key when available. NeMo Guardrails reads the
# standard OpenAI environment variable internally, so expose the same proxy key
# there as well.
config = load_app_config()
os.environ["OPENAI_API_KEY"] = config.litellm_api_key

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=config.openai_model,
    base_url=config.openai_base_url,
    api_key=config.litellm_api_key,
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model=config.openai_embeddings_model,
    base_url=config.openai_base_url,
    api_key=config.litellm_api_key,
    show_progress_bar=True
)

# Redis configuration
REDIS_URL = config.redis_url

# Initialize the Langfuse Cloud client with the configured project.
langfuse = initialize_langfuse_client(
    public_key=config.langfuse_public_key,
    secret_key=config.langfuse_secret_key,
    host=config.langfuse_host,
)


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed_documents")
@lru_cache(maxsize=1)
def embed_documents(json_path: str) -> QdrantVectorStore | list:
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        QdrantVectorStore | list: A Qdrant vector store built from the smartphone documents,
            or an empty list if an error occurs.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.exception("Smartphone data file was not found: %s", json_path)
        return []
    except json.JSONDecodeError:
        logger.exception("Could not decode smartphone data file: %s", json_path)
        return []
    except Exception:
        logger.exception("Unexpected error while reading smartphone data file: %s", json_path)
        return []

    documents = []
    for entry in data:
        # Build a readable content string from the JSON entry
        content = (
            f"Model: {entry.get('model', '')}\n"
            f"Price: {entry.get('price', '')}\n"
            f"Rating: {entry.get('rating', '')}\n"
            f"SIM: {entry.get('sim', '')}\n"
            f"Processor: {entry.get('processor', '')}\n"
            f"RAM: {entry.get('ram', '')}\n"
            f"Battery: {entry.get('battery', '')}\n"
            f"Display: {entry.get('display', '')}\n"
            f"Camera: {entry.get('camera', '')}\n"
            f"Card: {entry.get('card', '')}\n"
            f"OS: {entry.get('os', '')}\n"
            f"In Stock: {entry.get('in_stock', '')}"
        )
        documents.append(Document(page_content=content))

    try:
        return initialize_qdrant_store(
            documents=documents,
            embeddings_model=embeddings_model,
            qdrant_url=config.qdrant_url,
            qdrant_api_key=config.qdrant_api_key,
        )

    except Exception:
        logger.exception("Could not initialize the smartphone vector store")
        return []

# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.
    """
    try:
        product_db = embed_documents("datasets/smartphones.json")
        if not isinstance(product_db, QdrantVectorStore):
            return "The smartphone product database is currently unavailable."

        results = product_db.similarity_search(model, k=1)

        if not results:
            return "Could not find information for the specified model."

        return results[0].page_content

    except Exception:
        logger.exception("Smartphone information retrieval failed for model %s", model)
        return "The smartphone product database is currently unavailable."


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
@observe(name="generate_context")
def generate_context(ai_message: AIMessage, conversation: list, config: dict | None = None) -> None:
    """
    Process tool calls from the language model and append their responses as ToolMessage objects
    to the conversation history in place.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.
        conversation (list): The current conversation history (in-memory for this turn).
        config (dict | None): Optional configuration dictionary containing callbacks for tracing.

    :returns
        None. The conversation list is updated in place.
    """
    # construct the conversation history with the AI message containing tool calls
    conversation.append(ai_message)

    # Check if the AI message has any tool calls
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        conversation.append(
            AIMessage(
                content="No tool calls found. Please ensure the model is configured to use tools."
            )
        )

    try:
        # Process each tool call, invoke the appropriate tool, and append the result to the conversation
        # a message with tool calls is expected to be followed by tool responses
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                # Pass config with callbacks to ensure tool invocation is traced
                tool_output = smartphone_info_tool.invoke(tool_call, config=config)
                conversation.append(tool_output)

    except Exception:
        logger.exception("An error occurred while processing tool calls")
        conversation.append(
            AIMessage(
                content="An error occurred while processing tool calls."
            )
        )


# ---------------------------
# Chain and Guardrails Setup
# ---------------------------
tools = [smartphone_info_tool]
llm_with_tools = llm.bind_tools(tools)
context_lf_prompt = langfuse.get_prompt("context_system_prompt")
review_lf_prompt = langfuse.get_prompt("review_system_prompt")

context_prompt = ChatPromptTemplate.from_messages([
    *context_lf_prompt.get_langchain_prompt(),
    MessagesPlaceholder(variable_name="conversation"),
])
context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

review_prompt = ChatPromptTemplate.from_messages([
    *review_lf_prompt.get_langchain_prompt(),
    MessagesPlaceholder(variable_name="conversation"),
])
review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

trimmer = trim_messages(
    strategy="last",
    token_counter=llm,
    max_tokens=500,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)

context_chain = context_prompt | trimmer | llm_with_tools
review_chain = review_prompt | llm
guardrails_config = RailsConfig.from_path("config/")
input_rails = RunnableRails(
    guardrails_config,
    input_key="user_input",
)

langfuse_handler = CallbackHandler()


@app.post("/ask")
def ask(request: QueryRequest) -> dict[str, str]:
    try:
        redis_history = create_redis_chat_history(
            session_id=request.session_id,
            redis_url=REDIS_URL,
            ttl=3600,
        )

        conversation = list(redis_history.messages)
        user_message = HumanMessage(content=request.user_input)
        conversation.append(user_message)
        with langfuse.start_as_current_observation(
            as_type="span",
            name="user-query",
            input=request.user_input,
        ) as span:
            with propagate_attributes(
                session_id=request.session_id,
                user_id=request.user_id,
            ):
                validation_result = input_rails.rails.generate(
                    messages=[
                        {
                            "role": "user",
                            "content": request.user_input,
                        }
                    ],
                    options=GenerationOptions(
                        rails=["input"],
                        output_vars=[
                            "allowed",
                            "triggered_input_rail",
                            "bot_message",
                        ],
                    ),
                )

                validation_context = validation_result.output_data or {}
                rail_triggered = (
                    validation_context.get("allowed") is False
                    or bool(validation_context.get("triggered_input_rail"))
                )

                if rail_triggered:
                    rail_response = validation_result.response[0]["content"]

                    span.update(
                        output=rail_response,
                        metadata={
                            "triggered_input_rail": validation_context.get(
                                "triggered_input_rail"
                            )
                        },
                    )

                    return {"response": rail_response}

                ai_with_tools = context_chain.invoke(
                    {
                        "user_input": request.user_input,
                        "conversation": conversation,
                    },
                    config={
                        "run_name": "context",
                        "callbacks": [langfuse_handler],
                    },
                )

                generate_context(
                    ai_with_tools,
                    conversation,
                    config={
                        "callbacks": [langfuse_handler],
                    },
                )

                response = review_chain.invoke(
                    {
                        "user_id": request.user_id,
                        "user_input": request.user_input,
                        "conversation": conversation,
                    },
                    config={
                        "run_name": "final-response",
                        "callbacks": [langfuse_handler],
                    },
                )
            span.update(output=response.content)

        redis_history.add_message(user_message)
        redis_history.add_message(response)

        return {"response": response.content}

    except Exception as e:
        logger.exception("Failed to generate a response")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate a response.",
        ) from e


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
