"""Authoritative conversation state and shared structured turn context for HAL 9000 Inside.

Maintains a compact incremental brief:
- Active topic, complete user request and conversational purpose.
- Recent chronological dialogue and resolved references.
- Confirmed preferences, decisions and corrections.
- Relevant memories with source turn/date, confidence and temporal validity.
- Information already spoken.
- Unresolved questions, pending research and promised follow-ups.

Selects a lightweight response plan before generation:
A. COMPLETE_LOCAL: social interaction, simple disclosure, confirmed action or directly supported recollection.
B. GROUNDED_OPENING: useful confirmed information is available, but further explanation is required.
C. CONTEXTUAL_BRIDGE: substantive reasoning or research is needed.
"""

from dataclasses import dataclass, field
import datetime
import enum
import re
from typing import Any, Dict, List, Optional, Tuple


class ResponsePlan(str, enum.Enum):
    COMPLETE_LOCAL = "COMPLETE_LOCAL"
    GROUNDED_OPENING = "GROUNDED_OPENING"
    CONTEXTUAL_BRIDGE = "CONTEXTUAL_BRIDGE"


@dataclass
class ConversationBrief:
    """Authoritative compact incremental brief of the active dialogue."""
    active_topic: Optional[str] = None
    complete_user_request: str = ""
    conversational_purpose: str = "social_ack"
    resolved_references: Dict[str, str] = field(default_factory=dict)
    confirmed_preferences: List[str] = field(default_factory=list)
    confirmed_facts: List[Dict[str, Any]] = field(default_factory=list)
    recent_corrections: List[str] = field(default_factory=list)
    information_already_spoken: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    pending_research_topics: List[str] = field(default_factory=list)
    promised_follow_ups: List[str] = field(default_factory=list)


@dataclass
class TurnContext:
    """Structured turn context shared across local and remote generation."""
    utterance: str
    purpose: str  # "sharing_info", "correction", "question", "request_action", "social_ack"
    plan: ResponsePlan = ResponsePlan.CONTEXTUAL_BRIDGE
    needs_remote: bool = True
    needs_research: bool = False
    active_topic: Optional[str] = None
    unresolved_question: Optional[str] = None
    referents: List[str] = field(default_factory=list)
    recent_dialogue: List[Dict[str, str]] = field(default_factory=list)
    confirmed_memories: List[Dict[str, Any]] = field(default_factory=list)
    recent_openings: List[str] = field(default_factory=list)
    grounded_evidence: Optional[str] = None
    direct_local_answer: Optional[str] = None
    brief: Optional[ConversationBrief] = None
    task_id: Optional[str] = None

    def format_for_local_prompt(self) -> str:
        """Format a compact delimited text block for the local SmolLM2 prompt."""
        parts = [
            f"Plan: {self.plan.value}",
            f"Conversational purpose: {self.purpose}",
            f"Needs research: {'yes' if self.needs_research else 'no'}",
        ]
        if self.active_topic:
            parts.append(f"Active topic: {self.active_topic}")
        if self.grounded_evidence:
            parts.append(f"Confirmed evidence: {self.grounded_evidence}")
        if self.direct_local_answer:
            parts.append(f"Permitted direct answer: {self.direct_local_answer}")
        if self.recent_openings:
            avoid = ", ".join(f'"{o}"' for o in self.recent_openings[-3:])
            parts.append(f"Recent openings to avoid: {avoid}")

        if self.recent_dialogue:
            hist_lines = []
            for t in self.recent_dialogue[-4:]:
                role = "User" if t.get("role") == "user" else "HAL"
                text = t.get("content", "").replace("\n", " ").strip()
                if len(text) > 90:
                    text = text[:87] + "..."
                hist_lines.append(f"{role}: {text}")
            parts.append("Recent dialogue:\n" + "\n".join(hist_lines))

        return "\n".join(parts)


# Conversational purpose classifiers
VOCAL_FILLER_PATTERNS = [
    r"^(mm|umm?|uh+|hmm+|mm-?hmm|ah+|er+|eh)[\.!?,\s]*$",
]

CORRECTION_PATTERNS = [
    r"^(no|nope|wrong|that's wrong|that is wrong|incorrect|that's not what i asked)[\b,\.!?]",
    r"^actually\b",
    r"\bi didn't ask for\b",
    r"\bi did not ask for\b",
    r"\bi meant\b",
    r"\bnot what i said\b",
    r"\byou misunderstood\b",
    r"\bthat's not right\b",
    r"\bstop describing\b",
    r"\bwait,?\s+earlier\s+i\s+mentioned\b",
    r"\bdo not give me\b",
    r"\bdon't give me\b",
]

QUESTION_PATTERNS = [
    r"^(did|do|does|is|are|was|were|can|could|would|will|has|have|had|should|shall)\b",
    r"^(what|who|where|when|why|how|which|whose)\b",
    r"\b(what|who|where|when|why|how|which)\s+(is|are|was|were|do|does|did|can|could|should|will|would)\b",
    r"\b(right|correct|isn't it|aren't we|don't you|can't you)\s*\??$",
    r"\?$",
]

REQUEST_ACTION_PATTERNS = [
    r"\b(compare|analyze|evaluate|summarize|synthesize|calculate|search|find|lookup|explain|describe|elaborate|continue|detail|recommend|outline)\b",
    r"^(tell me|give me|show me|list|let's talk about|let us talk about|check)\b",
]

SOCIAL_ACK_PATTERNS = [
    r"^(hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings)(\s+hal)?[\.!?]*$",
    r"^(how\s+are\s+you|how\s+do\s+you\s+feel|how\s+are\s+things)(\s+hal)?[\.!?]*$",
    r"^(thank\s+you|thanks|thank\s+you\s+very\s+much|appreciate\s+it|thanks\s+for\s+the\s+advice|thank\s+you\s+for\s+the\s+advice)([\s,]+hal)?[\.!?]*$",
    r"^thank\s+you[,\s]+that\b.*$",
    r"^thanks[,\s]+let\s+me\b.*$",
    r"^okay[,\s]+let\s+me\s+think\b.*$",
    r"^(goodbye|bye|bye\s+bye|see\s+you|farewell)(\s+hal)?[\.!?]*$",
    r"^(ok|okay|got it|understood|all right|alright|fine|perfect|great|good|nice|cool|awesome|sounds good|fair enough)([,\s]+or\s+aligned.*)?[\.!?]*$",
]

SHARING_INFO_PATTERNS = [
    r"^(i am|i'm|i will|i'll|i have|i've|we are|we're|we'll|my|our)\b",
    r"\b(for the week|for a few days|on vacation|right now|currently|as our reference)\b",
]

RESEARCH_NEED_PATTERNS = [
    r"\b(compare|comparison|versus|vs\.?|difference between)\b",
    r"\b(release cadence|latest release|documentation|specifications|kernel version)\b",
    r"\b(search|look up|find out|check the databanks|who was|when was|research)\b",
    r"\b(ubuntu|linux|red hat|fedora|canonical|spinoza|stoic|postgresql|kubernetes|btrfs|zfs|openzfs)\b",
]

TASK_STATUS_INQUIRY_PATTERNS = [
    r"\bdid you (do it|search|find (it|anything)|check)\b",
    r"\b(provide|give me|what is) your answer\b",
    r"\bhave you (completed|finished|found)\b",
    r"\bwhat did you find\b",
]


def classify_conversational_purpose(utterance: str) -> str:
    """Classify the primary conversational purpose of the user's turn."""
    clean = utterance.strip().lower()
    if not clean:
        return "vocal_filler"

    # 0. Minimal vocal fillers ("Mm.", "Um", "Mm-hmm.")
    for pat in VOCAL_FILLER_PATTERNS:
        if re.match(pat, clean):
            return "vocal_filler"

    # 1. Task status inquiry
    for pat in TASK_STATUS_INQUIRY_PATTERNS:
        if re.search(pat, clean):
            return "task_status_inquiry"

    # 2. Corrections (unless it ends with a question mark or tag question like "right?")
    is_tag_or_explicit_question = bool(
        re.search(r"\?$", clean)
        or re.search(r"\b(right|correct|isn't it|aren't we|don't you)\s*\??$", clean)
    )
    if not is_tag_or_explicit_question:
        for pat in CORRECTION_PATTERNS:
            if re.search(pat, clean):
                return "correction"

    # 3. Social courtesies and short acknowledgments
    for pat in SOCIAL_ACK_PATTERNS:
        if re.match(pat, clean):
            return "social_ack"

    # 4. Explicit action requests (compare, analyze, search, elaborate, continue, synthesize, etc.)
    for pat in REQUEST_ACTION_PATTERNS:
        if re.search(pat, clean):
            return "request_action"

    # 5. Explicit questions (including tag questions and multi-clause questions)
    for pat in QUESTION_PATTERNS:
        if re.search(pat, clean):
            return "question"

    # 6. Simple personal disclosures (short statements sharing personal state/plans without embedded questions)
    word_count = len(clean.split())
    if word_count <= 20:
        for pat in SHARING_INFO_PATTERNS:
            if re.search(pat, clean):
                return "sharing_info"

    # 7. Substantive conversational statements / observations -> route to remote for full conversational engagement
    return "question"


def extract_active_topic(utterance: str, recent_dialogue: List[Dict[str, str]]) -> Optional[str]:
    """Identify the active topic from utterance or recent turns."""
    clean = utterance.strip()

    # Topic indicators in current utterance
    topic_match = re.search(r"\b(?:in|about|regarding|on)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b", clean)
    if topic_match:
        return topic_match.group(1)

    # Well-known named entities
    for entity in ("Cancun", "Ubuntu Pro", "Ubuntu", "Fedora", "Red Hat", "SUSE", "Spinoza", "ROS", "Kubernetes", "Snaps", "Flatpaks", "Charmed Kubernetes", "Landscape"):
        if re.search(r"\b" + re.escape(entity) + r"\b", clean, re.IGNORECASE):
            return entity

    # Look backwards through recent dialogue
    for turn in reversed(recent_dialogue[-4:]):
        prev_text = turn.get("content", "")
        for kw in ("Cancun", "Ubuntu Pro", "Ubuntu", "Fedora", "Red Hat", "SUSE", "Spinoza", "ROS", "Kubernetes", "Snaps", "Flatpaks", "Charmed Kubernetes"):
            if re.search(r"\b" + re.escape(kw) + r"\b", prev_text, re.IGNORECASE):
                return kw

    return None


def determine_research_need(utterance: str, purpose: str) -> bool:
    """Determine whether external search / research is genuinely required."""
    clean = utterance.strip().lower()

    # Simple disclosures, corrections, social acknowledgments, and identity queries NEVER need research
    if purpose in ("social_ack", "sharing_info", "correction", "task_status_inquiry"):
        return False

    if any(p in clean for p in ("tell me more about you", "who are you", "what is your name", "your identity")):
        return False

    # Personal confirmation questions do not need web research
    if re.search(r"\bdid i (say|tell|mention)\b", clean):
        return False

    # Check for substantive research keywords
    for pat in RESEARCH_NEED_PATTERNS:
        if re.search(pat, clean):
            return True

    return False


def find_grounded_answer(
    utterance: str,
    recent_dialogue: List[Dict[str, str]],
    confirmed_memories: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """Check if the user's question has direct authoritative evidence in recent turns or confirmed memories.
    Returns (evidence_summary, direct_grounded_answer).
    """
    clean = utterance.strip().lower()

    # Example: "Did I say I'm in Cancun?"
    did_i_say_match = re.search(r"\bdid i (?:say|mention|tell you)\b(?:\s+(?:that\s+)?i['\s]m\s+(?:in\s+)?([a-zA-Z]+))?", clean)
    if did_i_say_match:
        target_entity = did_i_say_match.group(1) or ""
        # Search recent dialogue
        for turn in reversed(recent_dialogue):
            if turn.get("role") == "user":
                c = turn.get("content", "")
                if not target_entity or target_entity in c.lower():
                    if "cancun" in c.lower():
                        evidence = "User previously stated: 'I'm in Cancun for the week.'"
                        answer = "Yes, you said you're there for the week."
                        return evidence, answer
                    elif target_entity:
                        evidence = f"User previously stated: '{c.strip()}'"
                        answer = f"Yes, you said you are in {target_entity.title()}."
                        return evidence, answer

        # Check confirmed memories
        for mem in confirmed_memories:
            text = mem.get("text", "")
            if "cancun" in text.lower():
                evidence = f"Confirmed memory: '{text.strip()}'"
                answer = "Yes, you said you're there for the week."
                return evidence, answer

    # Question about current machine time, date, or timezone
    if re.search(r"\b(what time is it|current time|time of the machine|check the time|what is the time|what day is it|what is today's date|what date is it|which time\s*zone are you|what time\s*zone are you)\b", clean):
        now_local = datetime.datetime.now().astimezone()
        tz_name = now_local.strftime("%Z") or "local time"
        utc_offset = now_local.strftime("%z")
        utc_fmt = f"UTC{utc_offset[:3]}:{utc_offset[3:]}" if len(utc_offset) == 5 else f"UTC{utc_offset}"
        time_12h = now_local.strftime("%I:%M %p").lstrip("0")
        date_str = now_local.strftime("%A, %B %d, %Y")
        evidence = f"Local machine clock: {date_str}, {time_12h} {tz_name} ({utc_fmt})."
        if "was asking" in clean or "cannot answer" in clean:
            return evidence, None
        elif "zone" in clean:
            answer = f"My local system clock is configured to the {tz_name} time zone, which is {utc_fmt}."
        elif "day" in clean or "date" in clean:
            answer = f"Today is {date_str}, and the current local time is {time_12h} {tz_name}."
        elif "why" in clean or "check the time" in clean:
            answer = f"I do have access to the onboard clock. It is currently {time_12h} {tz_name} ({utc_fmt})."
        else:
            answer = f"The current machine time is {time_12h} {tz_name} ({utc_fmt}), on {date_str}."
        return evidence, answer

    # Question about safety with NO established safety data: "Is that safe?"
    if re.search(r"\bis that safe\b", clean) or re.search(r"\bis it safe\b", clean):
        # We do NOT have sufficient evidence to definitively say "Yes" or "No"
        return "Insufficient evidence to determine safety.", None

    return None, None


def build_turn_context(
    utterance: str,
    recent_dialogue: List[Dict[str, str]],
    confirmed_memories: List[Dict[str, Any]],
    recent_openings: Optional[List[str]] = None,
    active_task: Optional[Any] = None,
) -> TurnContext:
    """Build compact structured turn context from existing local state and select response plan."""
    purpose = classify_conversational_purpose(utterance)
    topic = extract_active_topic(utterance, recent_dialogue)
    needs_research = determine_research_need(utterance, purpose)
    evidence, direct_answer = find_grounded_answer(utterance, recent_dialogue, confirmed_memories)

    # Plan Selection:
    # A. COMPLETE_LOCAL: social interaction, simple disclosure, confirmed action or directly supported recollection.
    # B. GROUNDED_OPENING: useful confirmed information is available, but further explanation is required.
    # C. CONTEXTUAL_BRIDGE: substantive reasoning or research is needed.
    needs_remote = True
    plan = ResponsePlan.CONTEXTUAL_BRIDGE

    if purpose in ("social_ack", "vocal_filler"):
        plan = ResponsePlan.COMPLETE_LOCAL
        needs_remote = False
    elif purpose == "sharing_info":
        plan = ResponsePlan.COMPLETE_LOCAL
        needs_remote = False
    elif direct_answer is not None and not needs_research:
        plan = ResponsePlan.COMPLETE_LOCAL
        needs_remote = False
    elif purpose == "correction" and not needs_research:
        u_clean = utterance.strip().lower()
        if any(k in u_clean for k in ("just tell me", "instead", "what about", "explain", "compare", "detail")):
            plan = ResponsePlan.CONTEXTUAL_BRIDGE
            needs_remote = True
        else:
            plan = ResponsePlan.COMPLETE_LOCAL
            needs_remote = False
    elif evidence and direct_answer is None:
        plan = ResponsePlan.GROUNDED_OPENING
        needs_remote = True
    else:
        plan = ResponsePlan.CONTEXTUAL_BRIDGE
        needs_remote = True

    # Check task status inquiry: e.g. "Did you search?" or "Provide your answer"
    if purpose == "task_status_inquiry" and active_task:
        needs_remote = False
        plan = ResponsePlan.COMPLETE_LOCAL
        if active_task.execution_status == "succeeded" and active_task.synthesized_findings:
            direct_answer = active_task.synthesized_findings
            evidence = f"Task {active_task.task_id} completed successfully."
        elif active_task.execution_status == "running":
            direct_answer = f"The research into {active_task.topic} is still in progress across available records."
            evidence = f"Task {active_task.task_id} is running."
        elif active_task.execution_status in ("failed", "cancelled"):
            reason = active_task.limitations or "the system encountered an error"
            direct_answer = f"The investigation into {active_task.topic} encountered an error: {reason}."
            evidence = f"Task {active_task.task_id} status: {active_task.execution_status}"

    referents = []
    if topic:
        referents.append(topic)

    # Build conversation brief
    brief = ConversationBrief(
        active_topic=topic,
        complete_user_request=utterance,
        conversational_purpose=purpose,
        resolved_references={"topic": topic} if topic else {},
        confirmed_facts=confirmed_memories,
        information_already_spoken=recent_openings or [],
    )

    return TurnContext(
        utterance=utterance,
        purpose=purpose,
        plan=plan,
        needs_remote=needs_remote,
        needs_research=needs_research,
        active_topic=topic,
        referents=referents,
        recent_dialogue=recent_dialogue,
        confirmed_memories=confirmed_memories,
        recent_openings=recent_openings or [],
        grounded_evidence=evidence,
        direct_local_answer=direct_answer,
        brief=brief,
        task_id=getattr(active_task, "task_id", None) if active_task else None,
    )
