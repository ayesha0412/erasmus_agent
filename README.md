# Erasmus Mundus Scholarship Tracker Agentic Workflow

Finding the right Erasmus Mundus programme is harder than it should be. This agent makes it easier, and then keeps helping you all the way to your applications.

## The problem

The [official catalogue](https://www.eacea.ec.europa.eu/scholarships/erasmus-mundus-catalogue_en) lists 218 programmes, but only by name. To find out what a programme actually teaches, who can apply and when the deadline is, you have to open each programme's own website, one by one.

Names don't help much either. A programme can teach machine learning and data engineering without "AI" ever appearing in its title. The only way to know is to read its course list.

And this isn't a one-time job. Every year, from about October to January, you check pages again, deadlines move, and you collect your documents bit by bit.

## What this agent does

You tell it about yourself once. It reads the programme pages for you, finds the ones that fit, and keeps track of what you still need to do.

Here's the flow:

1. **You create your profile.** Your degree, grades, English test score, interests and the documents you already have. You never type this again.
2. **The agent reads each programme page.** It turns the messy webpage into a clear record: who can apply, deadlines, required documents, and the full course list.
3. **Every course gets subject tags.** For example, "Machine Learning" is tagged *artificial intelligence*. Now a programme matches your interest in AI even if its name never says so.
4. **Your eligibility is checked.** Plain code compares your profile with each programme and gives one of three results: Eligible, Not eligible or Uncertain, always with the reason.
5. **You approve a shortlist.** The agent suggests the best matches. Nothing is saved until you say yes.
6. **You get your to-do list.** For each shortlisted programme, the agent lists the documents you're missing and writes them to an Excel checklist and a calendar of deadlines.

## It remembers

The next time you run it, even after closing the program, it picks up where you left off:

- If a programme changes its deadline, the agent notices, asks you to confirm, and updates your plan. The old date is kept in history.
- When you add a new document, the agent ticks it off your checklist.
- If you say "no Law programmes", it won't suggest them again.

Everything is stored in one small SQLite database file. The Excel checklist and calendar are created from it, so they always match.

## The AI models

You can switch between three open models with one setting:

| Role | Model | Where it runs |
| --- | --- | --- |
| Main | Qwen3.8 27B | Groq (cloud, free plan) |
| Switch-in | GPT-OSS-120B | Groq (cloud) |
| Offline backup | Qwen3.5 9B | Your own computer, via Ollama |

The AI reads the pages and decides the next step. Plain code does the checking: eligibility, dates and saving records. That way the important decisions are always exact.

## Getting started

You'll need **Python 3.12** and **git**. Ollama is only needed if you want the offline backup.

**1. Download the project**

```powershell
git clone https://github.com/<your-username>/erasmus-mundus-tracker.git
cd erasmus-mundus-tracker
```

**2. Create and activate a virtual environment**

```powershell
python -m venv erasmusvenv
erasmusvenv\Scripts\Activate.ps1
```

**3. Install the packages**

```powershell
pip install -r requirements.txt
```

**4. Add your settings**

```powershell
copy .env.example .env
```

Open `.env` and paste in your Groq API key. You can get one for free at [console.groq.com](https://console.groq.com/keys).

How to run the agent will be added here as it is built.

## What's inside

| Folder | What it holds |
| --- | --- |
| `erasmus_agent/` | The code: agent, tools, memory and command-line app |
| `data/` | Sample student profiles and saved programme pages |
| `tests/` | Tests and the evaluation scenarios |
| `outputs/` | Generated checklists, calendars and results |
| `logs/` | A record of every action the agent takes |

## Progress

- [ ] M0 Setup and model switch
- [ ] M1 Sample data
- [ ] M2 Database and version history
- [ ] M3 Data models
- [ ] M4 Tool 1: read a programme page
- [ ] M5 AI fills in programme records
- [ ] M6 Tool 2: profile lookup and eligibility checks
- [ ] M7 The agent loop
- [ ] M8 Approval and Tool 3: write the plan
- [ ] M9 Command-line app
- [ ] M10 Restart, cycles 2 and 3
- [ ] M11 Action log
- [ ] M12 Failures and safe retries
- [ ] M13 Safety and privacy between students
- [ ] M14 Ten test scenarios
- [ ] M15 Memory on vs off comparison
- [ ] M16 Performance figures
- [ ] M17 Packaging
- [ ] M18 Report
- [ ] M19 Demo

## About this project

This is an individual assignment for *Practical Agentic Systems with Persistent Memory* (catalogue option A08, adapted). All student data in this repo is made up. Programme pages are saved copies of public websites, used only for testing.

Parts of the design and code were developed with help from Claude (Anthropic). All code has been reviewed and is understood by the author.