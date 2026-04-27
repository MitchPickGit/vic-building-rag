# Project setup — read this first

One-time setup to get Claude Code ready to work on this project.

## Step 1 — Install Claude Code

If you don't have it yet:

```bash
npm install -g @anthropic-ai/claude-code
```

Requires Node.js. If you don't have Node, install it first via nvm or your
OS package manager.

## Step 2 — Create the project directory

```bash
mkdir -p ~/projects/building-code-app
cd ~/projects/building-code-app
```

## Step 3 — Copy the briefing files in

From this package, copy these files into the project directory:

```
README.md
CLAUDE.md
NEXT_STEPS.md
PARSER_DESIGN.md
FIRST_SESSION.md
requirements.txt
.gitignore
parse_building_act.py
building_act_chunks.jsonl
test_questions.json
evaluate.py
```

Then create these folders:

```bash
mkdir -p docs
```

## Step 4 — Download the source PDFs

Place the Building Act PDF (the one we've been working with) at:
```
docs/Building_Act_1993.pdf
```

Download the Building Regulations 2018 and place at:
```
docs/Building_Regulations_2018.pdf
```

Get it from https://www.legislation.vic.gov.au/ and search for "Building
Regulations 2018". Use the authorised consolidated PDF, not the
as-made version.

These PDFs are gitignored so you'll need to re-download if you clone the
repo on another machine.

## Step 5 — Initialise git

```bash
git init
git add .
git commit -m "chore: initial project scaffolding from briefing"
```

## Step 6 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 7 — Set API keys

Create a `.env` file (gitignored) in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
```

Get the Anthropic key from https://console.anthropic.com.
Voyage (for embeddings) from https://www.voyageai.com. Free tier is enough
to start. OpenAI embeddings work equivalently if you already have an
OpenAI key — just use OPENAI_API_KEY instead.

## Step 8 — Sanity check

```bash
python evaluate.py
```

You should see:
```
hit@1: 37%  (11/30)
hit@3: 53%  (16/30)
hit@5: 63%  (19/30)
```

If you get errors, the environment isn't set up correctly — fix that
before involving Claude Code.

## Step 9 — Start Claude Code

```bash
cd ~/projects/building-code-app
claude
```

Then paste the kickoff prompt from `FIRST_SESSION.md`.

## Step 10 — After your first session

Whatever Claude Code builds, review the diff before committing. Read the
code. If something looks wrong or surprising, ask about it — don't just
accept it. Your ability to understand the code you're shipping is what
makes this a real product rather than a black box.
