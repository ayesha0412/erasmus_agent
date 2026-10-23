# Erasmus Mundus Scholarship Tracker

An AI agent that helps a student find Erasmus Mundus master's programmes they are eligible for and keep track of what they still need to apply. It works in repeated cycles across an application season and remembers everything between runs, so each visit builds on the last one.

## The problem

The official Erasmus Mundus catalogue lists around 218 joint master's programmes, but only by name, twenty to a page. The details that matter (what is taught, who can apply, the deadline, which documents are needed) are spread across each programme's own website. Names can also be misleading: a programme may teach machine learning without "AI" anywhere in its title.

The search is not a one-off job. Applications open around October and close between December and January, deadlines change, and a student gathers documents bit by bit over several weeks. Doing this by hand means checking the same pages again and again and keeping notes that quickly go out of date.

## What the agent does

1. **Gets to know you.** You enter your degree, grades, English test score, interests and the documents you already have.
2. **Reads programme pages.** It reads saved copies of each programme's web page.
3. **Pulls out the key facts.** An AI model turns each page into a structured record: entry requirements, deadlines, required documents and the list of courses.
4. **Checks eligibility.** Plain code compares your profile with each programme's requirements and says Eligible, Not eligible or Uncertain, always with reasons.
5. **Ranks by fit.** Programmes you are eligible for are ranked by how many of their courses match your interests.
6. **Decides what to do next.** Based on what it has found, the AI chooses the next step, for example asking you a question when a page is vague, or asking for your approval when a deadline has changed.
7. **Asks before it acts.** You approve the shortlist. Nothing is written until you say yes.
8. **Builds your to-do list.** It works out which documents you are still missing and creates an Excel checklist and a calendar file with the deadlines.

## It remembers

All of this is stored in a local database, so the agent picks up where it left off, even after the program is closed and restarted. In a later cycle it is designed to:

- **notice changed deadlines** by spotting that a page has changed since the last visit, keep the old date as history, and ask you before using the new one;
- **take in new documents**, such as an updated English certificate or a new recommendation letter, and tick them off the checklist;
- **respect your preferences**, such as "exclude law programmes", from then on;
- **keep a full history**: corrections add a new version and never delete the old one, so you can always see what changed and when.

## Where the data comes from

| Data | Type | Notes |
|-|-|-|
| Programme catalogue (names and links) | Real | Collected once from the public Erasmus Mundus catalogue |
| Five programme pages | Real | Saved once from each programme's public website, with the address, date and a fingerprint |
| Your student profile | Entered by you | Typed in during onboarding and stored only on your computer |
| Sample students (Sara, Omar) | Fictional | Used only for tests and the demo |
| Two test copies of a page | Real page with one change | One has a revised deadline, one contains a planted instruction, to test how the agent reacts |

Pages are collected politely: the scraper respects each site's `robots.txt`, waits about two seconds between requests, identifies itself as a student project, and saves copies so the agent never needs to visit the live sites again.

## How matching works

**Eligibility and fit are kept separate.**

- **Eligibility** depends only on entry requirements: your degree, grades, English score and the deadline. It is decided by plain code, never by the AI.
- **Fit** shows how well a programme matches your interests. It only ranks programmes you are already eligible for.

Fit uses **tags**. Each course is labelled from a fixed list of topics, and each topic can point to a broader one. For example:

```
Course "Statistical Learning"  →  tag: machine learning
machine learning               →  broader: artificial intelligence
```

If your interest is *artificial intelligence*, this course counts as a match, even though neither the course nor the programme mentions AI by name. The agent then shows its evidence, for example: "4 of 12 courses cover artificial intelligence: Statistical Learning, …". Some courses, such as Academic Writing, have no tags at all, which is fine.

## The AI models

The agent uses open models hosted by [Groq](https://groq.com):

- **Main model:** `openai/gpt-oss-120b`, for reading pages and deciding the next step.
- **Fallback model:** `openai/gpt-oss-20b`, smaller and faster, used when the main one is busy or unavailable.

**What the AI does:** reads messy web pages and turns them into structured records, labels courses with tags from the fixed list, suggests tags for your interests, and chooses the next action based on what it has found.

**What plain code does:** everything that must be exact and repeatable, such as eligibility checks, match scores, missing-document lists, saving records and version history, and enforcing approvals. Text on a web page is treated as data, never as instructions to the agent.

## Getting started

You need **Python 3.12** and a free **Groq API key**. These steps are for Windows PowerShell.

1. Download the project:

   ```powershell
   git clone https://github.com/<your-username>/erasmus_agent.git
   cd erasmus_agent
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv erasmusvenv
   erasmusvenv\Scripts\Activate.ps1
   ```

   If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then activate again.

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Create your settings file from the template:

   ```powershell
   copy .env.example .env
   ```

5. Get a key at [console.groq.com/keys](https://console.groq.com/keys) and paste it into `.env` after `GROQ_API_KEY=`. Keep `.env` private; it is already excluded from git.

6. Check that the models respond:

   ```powershell
   python -m erasmus_agent.llm
   ```

   Both models should reply "pong".

### How to run

The command-line commands will be added here as the CLI is built.

## What's inside

| Folder | What it holds |
|-|-|
| `erasmus_agent/` | The agent's source code: AI client, settings, tools, memory and command line |
| `scripts/` | One-off helper scripts, such as the scraper that collected the programme pages |
| `data/` | All input data the agent reads |
| `data/pages/real/` | Saved copies of the five real programme pages, with their metadata |
| `data/pages/test/` | The two test copies (changed deadline, planted instruction) |
| `data/profiles/` | Fictional sample student profiles used for tests |
| `data/cycles/` | Inputs for later cycles: dates, new documents, new preferences |
| `data/private/` | Real personal profiles; never committed to git |
| `tests/` | Automated tests |
| `outputs/` | Generated checklists, calendar files and evaluation results |
| `logs/` | Records of every tool call, memory read and write, and error |

## About this project

This is an individual assignment for the course *Practical Agentic Systems with Persistent Memory* (catalogue option A08, adapted). Programme information was collected from public websites for study purposes only. All student data used in tests and the demo is fictional.

Parts of the design and code were developed with help from an AI assistant; all code was written, run and reviewed by the author, who understands and can modify all of it.