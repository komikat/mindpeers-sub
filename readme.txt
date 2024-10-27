===
Hindustan Consultancy Services
Mental Health Analyzer
built on OpenAI gpt4o-mini
===

Filetree
.
├── app.py                # the flask app used for user input
├── db.py                 # script for initializing sqlite3 db
├── gpt.py                # scripts for chain of thought and few shot prompting gpt4o-mini
├── inference.py          # inference of the older llama3.2 and sentence transformers based models
├── pyrightconfig.json    
├── readme.txt            # the file you are currently reading
├── sheet.tsv             # sample spreadsheet converted to tab seperated values
├── static
│   └── grid.svg
├── templates
│   └── home.html
├── train.py              # contains the fine-tune loop for older concern classification model
└── users.db              # db file populated on running the server

Method
- we first used a sentence transformer model and fine tuned it on classification task (+ used a llama3.2-3b Instruct model for other categories)
  using the spreadsheet provided. It was only able to correctly answer one single concern
  which is why we moved on to prompt engineering a much powerful gpt4o model.
- we are using chain of thought and few shot prompting on gpt-4o-mini.
  - provide it a step by step way to reason about the problem
  - ask it to return the answer in a structured 4-tuple (s1, s2, s3, s4) where s_i represents the i_th step of reasoning
  - we parse the structured tuple for displaying the results.
  - we pass the last five status updates with a prompt to use the 4-tuples for inferring the change in emotions over time.
- a demo video has been uploaded to youtube: https://www.youtube.com/watch?v=7kAUg2A935M

Instructions on getting it to run
- get an openai key and add it to your environment first (you can add this to your ~/.bashrc)
export OPENAI_API_KEY="sk-proj..."
- create a venv (optional but recommended)
python3 -m venv .whatever
source .whatever/bin/activate
- install dependencies
pip install openai flask
- run the flask server
python3 app.py
- open localhost:8123 on your browser

===
Akshit Kumar, Anushka Jain
===
