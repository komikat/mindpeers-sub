import ast

from openai import OpenAI


def isevaluatable(s):
    try:
        ast.literal_eval(s)
        return True
    except ValueError:
        return False


def pipeline(client: OpenAI, s):
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a helpful mental health assistant
            created to analyse complex human emotions. You will be
            given a user input and you need to analyse it step by
            step in the following way, selecting *exactly* and *only* the closest matching options from the ones given in between ().
            Step 1: Identify emotion polarity that is what you feel the emotion is about the text. If there are conflicting emotions declare neutral. (Positive, Negative, Neutral) 
            Step 2: Extract all concerns in the given text. (feeling hopeful, happy and excited, constantly worried, feeling very anxious, feeling much better, not eating properly, worried about health, confused about job prospects, extremely stressed, can't sleep well, feeling very low)
            Step 3: Identify categories of all concerns identified in previous step. (Positive outlook, Stress, Insomnia, Depression, Anxiety, Career Confusion, Health Anxiety, Eating Disorder)
            Step 4: Identify intensity of each concern identified in previous step, a 0 indicated no concern, a 3 indicating slight concern, a 5 indicating medium concern, a 7 indicating serious and a 10 indicated extreme concern. (0 - 10)
            
            Answer in a 4 tuple (s1, s2, s3, s4) where each s-tuple consist of a list of all identified options.
            ***
            Example:
            User: I can’t sleep well and I feel very low."
            Output: ("Negative", ["can't sleep well", "feel very low"], ["Insomnia", "Depression"], [6, 7])
            ***
            """,
            },
            {
                "role": "user",
                "content": s,
            },
        ],
        model="gpt-4o-mini",
    )

    res = completion.choices[0].message.content
    if isevaluatable(res):
        return completion.choices[0].message.content
    else:
        return pipeline(client, s)


def timeline(client: OpenAI, s):
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant expert in analysing human emotions over time and identify the shifts.
                You will be given a list of 4-tuples and you're tasked to analyse the progression of emotions and generate one short sentence summarising your analysis.
                The first element of each tuple represents the polarity of emotion, the second element represents the list of extracted concerns from the past, the third elements represents the category of the concerns and the last tuple represents the intensity of each concern.
                ***
                Example:
                User: [("Negative", ["can't sleep well", "feel very low"], ["Insomnia", "Depression"], [6, 7]), ("Positive", ["still anxious"], ["Anxiety"], [4])
                Output: 'Signs of improvement from Depression to Anxiety.'
                ***
                """,
            },
            {"role": "user", "content": s},
        ],
        model="gpt-4o-mini",
    )

    return completion.choices[0].message.content


"""
a1 = pipeline("I'm very scared about my job prospects.")
a2 = pipeline("I feel calm and relaxed, I think im hopeful.")
print(timeline(f"{[str(a1), str(a2)]}"))
"""
