import httpx
import asyncio
import json
import re
from typing import List, Optional, TypeVar, Dict, Union
from pydantic import BaseModel
from litellm import acompletion

from reag.prompt import REAG_SYSTEM_PROMPT
from reag.schema import ResponseSchemaMessage
from reag.memory import Memory


class Document(BaseModel):
    name: str
    content: str
    metadata: Optional[Dict[str, Union[str, int]]] = None


class MetadataFilter(BaseModel):
    key: str
    value: Union[str, int]
    operator: Optional[str] = None


T = TypeVar("T")


class QueryResult(BaseModel):
    content: str
    reasoning: str
    is_irrelevant: bool
    document: Document


class StreamingQueryResult(BaseModel):
    content: str
    reasoning: Optional[str] = None
    is_irrelevant: Optional[bool] = None
    document: Optional[Document] = None


DEFAULT_BATCH_SIZE = 20


class ReagClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system: str = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        schema: Optional[BaseModel] = None,
        model_kwargs: Optional[Dict] = None,
        memory: Optional[Memory] = None,
    ):
        self.model = model
        self.system = system or REAG_SYSTEM_PROMPT
        self.batch_size = batch_size
        self.schema = schema or ResponseSchemaMessage
        self.model_kwargs = model_kwargs or {}
        self.memory = memory or Memory()
        self._http_client = None

    async def __aenter__(self):
        self._http_client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._http_client:
            await self._http_client.aclose()

    def _filter_documents_by_metadata(
        self, documents: List[Document], filters: Optional[List[MetadataFilter]] = None
    ) -> List[Document]:
        if not filters:
            return documents

        def passes_filter(doc: Document, f: MetadataFilter) -> bool:
            if not doc.metadata or f.key not in doc.metadata:
                return False

            val = doc.metadata[f.key]
            op = f.operator or "equals"

            if op == "equals":
                return val == f.value
            if op == "notEquals":
                return val != f.value
            if op == "contains":
                return isinstance(val, str) and f.value in val
            if op == "startsWith":
                return isinstance(val, str) and val.startswith(f.value)
            if op == "endsWith":
                return isinstance(val, str) and val.endswith(f.value)
            if op == "regex":
                return isinstance(val, str) and re.match(f.value, val) is not None
            if op == "greaterThan":
                return val > f.value
            if op == "lessThan":
                return val < f.value
            if op == "greaterThanOrEqual":
                return val >= f.value
            if op == "lessThanOrEqual":
                return val <= f.value
            return False

        return [
            doc for doc in documents if all(passes_filter(doc, f) for f in filters)
        ]

    def _extract_think_content(self, text: str) -> tuple[str, str, bool]:
        """Extract content from think tags and parse the bulleted response format."""
        # Extract think content
        think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        
        # Remove think tags and get remaining text
        remaining_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # Initialize default values
        content = ""
        is_irrelevant = True
        
        # Extract is_irrelevant value
        irrelevant_match = re.search(r'\*\*isIrrelevant:\*\*\s*(true|false)', remaining_text, re.IGNORECASE)
        if irrelevant_match:
            is_irrelevant = irrelevant_match.group(1).lower() == 'true'
        
        # Extract content value
        content_match = re.search(r'\*\*Answer:\*\*\s*(.*?)(?:\n|$)', remaining_text, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
        
        return content, reasoning, is_irrelevant

    async def query(
        self,
        prompt: str,
        documents: List[Document],
        options: Optional[Dict] = None,
        stream: bool = False,
    ):
        if stream:
            return self._streaming_query(prompt, documents, options)
        else:
            return await self._batch_query(prompt, documents, options)

    async def _batch_query(
        self, prompt: str, documents: List[Document], options: Optional[Dict] = None
    ) -> List[QueryResult]:
        # Batch query logic remains the same
        try:
            # Convert dictionary filters to MetadataFilter objects
            filters = None
            if options and "filter" in options:
                raw_filters = options["filter"]
                if isinstance(raw_filters, list):
                    filters = [
                        MetadataFilter(**f) if isinstance(f, dict) else f
                        for f in raw_filters
                    ]
                elif isinstance(raw_filters, dict):
                    filters = [MetadataFilter(**raw_filters)]

            filtered_documents = self._filter_documents_by_metadata(documents, filters)

            def format_doc(doc: Document) -> str:
                return f"Name: {doc.name}\nMetadata: {doc.metadata}\nContent: {doc.content}"

            batch_size = self.batch_size
            batches = [
                filtered_documents[i : i + batch_size]
                for i in range(0, len(filtered_documents), batch_size)
            ]

            results = []
            for batch in batches:
                tasks = []
                # Create tasks for parallel processing within the batch
                for document in batch:
                    system = f"{self.system}\n\n# Available source\n\n{format_doc(document)}"
                    history = self.memory.get_history()
                    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": prompt}]
                    tasks.append(
                        acompletion(
                            model=self.model,
                            messages=messages,
                            response_format=self.schema,
                            **self.model_kwargs,
                        )
                    )

                # Process all documents in the batch concurrently
                batch_responses = await asyncio.gather(*tasks)

                # Process the responses
                for document, response in zip(batch, batch_responses):
                    message_content = response.choices[0].message.content

                    try:
                        if self.model.startswith("ollama/"):
                            content, reasoning, is_irrelevant = self._extract_think_content(message_content)
                            results.append(
                                QueryResult(
                                    content=content,
                                    reasoning=reasoning,
                                    is_irrelevant=is_irrelevant,
                                    document=document,
                                )
                            )
                            self.memory.add_message({"role": "user", "content": prompt})
                            self.memory.add_message({"role": "assistant", "content": content})
                        else:
                            # Ensure it's parsed as a dict
                            data = (
                                json.loads(message_content)
                                if isinstance(message_content, str)
                                else message_content
                            )

                            if data["source"].get("is_irrelevant", True):
                                continue

                            content = data["source"].get("content", "")
                            self.memory.add_message({"role": "user", "content": prompt})
                            self.memory.add_message({"role": "assistant", "content": content})

                            results.append(
                                QueryResult(
                                    content=content,
                                    reasoning=data["source"].get("reasoning", ""),
                                    is_irrelevant=data["source"].get("is_irrelevant", False),
                                    document=document,
                                )
                            )
                    except json.JSONDecodeError:
                        print("Error: Could not parse response:", message_content)
                        continue

            return results

        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")

    async def _streaming_query(
        self, prompt: str, documents: List[Document], options: Optional[Dict] = None
    ):
        # Convert dictionary filters to MetadataFilter objects
        filters = None
        if options and "filter" in options:
            raw_filters = options["filter"]
            if isinstance(raw_filters, list):
                filters = [
                    MetadataFilter(**f) if isinstance(f, dict) else f
                    for f in raw_filters
                ]
            elif isinstance(raw_filters, dict):
                filters = [MetadataFilter(**raw_filters)]

        filtered_documents = self._filter_documents_by_metadata(documents, filters)

        def format_doc(doc: Document) -> str:
            return f"Name: {doc.name}\nMetadata: {doc.metadata}\nContent: {doc.content}"

        system = f"{self.system}\n\n# Available source\n\n{format_doc(filtered_documents[0])}"
        history = self.memory.get_history()
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": prompt}]

        response_stream = await acompletion(
            model=self.model,
            messages=messages,
            stream=True,
            **self.model_kwargs,
        )

        full_response = ""
        async for chunk in response_stream:
            chunk_content = chunk.choices[0].delta.content or ""
            full_response += chunk_content
            yield StreamingQueryResult(content=chunk_content)

        self.memory.add_message({"role": "user", "content": prompt})
        self.memory.add_message({"role": "assistant", "content": full_response})
