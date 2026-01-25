
from typing_extensions import Literal
from pydantic import BaseModel, validator

class ConceptExtractionOutput(BaseModel):
    concepts: list[str]

class ChunkClassification(BaseModel):
    "YES or NO if a chunk contains important concepts."
    relevance: Literal["YES", "NO"]

class RoleTaggerOutput(BaseModel):
    "Role for concept in a chunk + a short evidence snippet."
    role: Literal["Example", "Definition", "Assumption"]
    snippet: str

ROLE_CLASSIFICATION_PROMPT = """
You will be given a text chunk and a concept from a university course lecture.
Classify the relevance of the concept to the chunk into one of three categories: Example, Definition, or Assumption.

Also return an evidence snippet copied from the chunk (10–30 words) that best supports your relevance label.
- The snippet MUST be an exact substring from the chunk.
- Keep it short.

Return strict JSON:
{ "role": "Example" | "Definition" | "Assumption", "snippet": "..." }
""".strip()


CHUNK_CLASSIFICATION_PROMPT = """
You are an academic course content classifier.

You will recieve a chunk extraced from a course material, you need to classify whether this text chunk is
ACADEMICALLY RELEVANT (contains course content, learning objectives, or topics)
or ADMINISTRATIVE/IRRELEVANT (contains logistics like office hours, grading policy, contact info, artifacts, unnecessary details that are not concepts).

Rules:
- YES → if it discusses course concepts, learning objectives, or knowledge areas.
- NO  → if it includes dates, locations, instructor info, policies, Zoom links, or grading tables, artifacts, or unnecessary details that are not concepts.

Return strict JSON:
{ "relevance": "YES" | "NO" }
""".strip()


CONCEPT_EXTRACTION_PROMPT = """
You are an instructor that extracts learning concepts from text. 

- Concept: a core idea or topic about the subject matter. Only extract meaningful course concepts.
DO NOT extract example values, variable names, numbers, formulas, or code elements.
Ignore content inside examples, formulas.


Return strict JSON:
{ "concepts": ["..."] }

Rules:
- 1–5 words per concept
- No code tokens, variable names, numbers, example values
- Deduplicate
""".strip()
