import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─────────────────────────────────────────────────────────────
# TEMPLATE PROMPTS — Each tuned for maximum engagement
# Target: 120-150 words = ~45-60 seconds at natural speaking pace
# ─────────────────────────────────────────────────────────────

TEMPLATES = {

    "reddit_story": """You are a viral Reddit story writer for YouTube Shorts and TikTok.

Write a first-person Reddit-style story about: {topic}

STRICT RULES:
- Start with a hook that creates immediate curiosity. Examples: "I accidentally discovered my neighbor's secret.", "My coworker tried to get me fired. It backfired spectacularly."
- Write in natural, conversational spoken English — no formal writing
- Include a twist or satisfying ending
- Exactly 120-150 words. Count carefully.
- NO hashtags, NO emojis, NO stage directions, NO "narration:", NO headers
- Write ONLY the script text that will be spoken. Nothing else.
- End with a punchy final line that makes people want to comment

Topic: {topic}

Write only the spoken script:""",

    "random_facts": """You are a viral facts content writer for YouTube Shorts and TikTok.

Write a "shocking facts" short about: {topic}

STRICT RULES:
- Open with: "You won't believe these facts about [topic]..." or similar
- Give exactly 5 facts, each more surprising than the last
- Each fact is 1-3 short sentences max
- End with the most mind-blowing fact last
- Use simple, punchy language — like you're telling a friend
- Exactly 120-150 words. Count carefully.
- NO hashtags, NO emojis, NO stage directions, NO numbering like "Fact 1:"
- Write ONLY the spoken script text. Nothing else.

Topic: {topic}

Write only the spoken script:""",

    "motivational": """You are a top-performing motivational content creator for YouTube Shorts and TikTok.

Write a motivational short about: {topic}

STRICT RULES:
- Open with a bold, provocative statement that stops the scroll. Do NOT start with "Are you..." or generic openers.
- Tell a brief, relatable story or scenario (2-3 sentences)
- Deliver the core insight or mindset shift clearly
- End with a powerful call to action or closing line people will screenshot
- Conversational, direct tone — like talking to a friend
- Exactly 120-150 words. Count carefully.
- NO hashtags, NO emojis, NO stage directions
- Write ONLY the spoken script text. Nothing else.

Topic: {topic}

Write only the spoken script:""",

    "news_summary": """You are a viral news summarizer for YouTube Shorts and TikTok.

Write a punchy news summary short about: {topic}

STRICT RULES:
- Open with "Breaking:" or "This just happened:" or a similar urgent hook
- Summarize the key facts in plain English — no jargon
- Explain WHY this matters to the average person
- End with a question or thought that drives comments: "What do you think about this?"
- Neutral, journalistic but conversational tone
- Exactly 120-150 words. Count carefully.
- NO hashtags, NO emojis, NO stage directions
- Write ONLY the spoken script text. Nothing else.

Topic: {topic}

Write only the spoken script:""",

    "custom": """You are an expert short-form video script writer for YouTube Shorts and TikTok.

The user has provided their own script or topic below. Your job is to:
1. If it's a complete script — clean it up for spoken delivery, fix pacing, make it punchy
2. If it's just a topic/idea — write a full engaging script about it

STRICT RULES:
- Natural spoken English, conversational tone
- Strong opening hook, clear middle, memorable ending
- Exactly 120-150 words
- NO hashtags, NO emojis, NO stage directions
- Write ONLY the spoken script text. Nothing else.

User input: {topic}

Write only the spoken script:""",
}


def generate_script(template: str, topic: str) -> str:
    """
    Generates a viral short-form video script using Gemini.
    Returns clean script text ready for voice synthesis.
    """
    if template not in TEMPLATES:
        template = "custom"

    prompt = TEMPLATES[template].format(topic=topic)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.8,   # Higher = more creative scripts
            max_output_tokens=500,
        ),
    )

    script = response.text.strip()

    # Clean up any accidental markdown or stage directions
    script = re.sub(r'\*\*.*?\*\*', '', script)        # Remove **bold**
    script = re.sub(r'\*.*?\*', '', script)             # Remove *italic*
    script = re.sub(r'#.*?\n', '', script)              # Remove # headers
    script = re.sub(r'\[.*?\]', '', script)             # Remove [stage directions]
    script = re.sub(r'\(.*?\)', '', script)             # Remove (parenthetical)
    script = re.sub(r'Narration:|Script:|Voiceover:', '', script, flags=re.IGNORECASE)
    script = re.sub(r'\n{3,}', '\n\n', script)         # Max 2 newlines
    script = script.strip()

    return script