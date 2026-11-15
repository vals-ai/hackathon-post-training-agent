INSTRUCTIONS_PROMPT = """
You will solve a competitive programming problem.

Write your solution as a complete C++20 program that reads standard input and writes standard output. Enclose code in three backticks, for example:

```
[your code here]
```

You have two tools. cpp_executor compiles a program and runs it on stdin you choose, for experiments, brute forces and stress tests. submission saves a solution as your answer and runs it on the sample tests from the statement. Your last submission that compiles is graded on hidden tests, under the time limit, with partial credit for each test it passes.

You may make at most 50 submissions in at most 100 turns. When you are done and do not want to submit again, respond with 'EXIT'.

The problem statement follows:
{question}
"""


def simple_extract_code(model_output: str):
    """
    Extract code from model output by finding the last code block between triple backticks.

    Args:
        model_output (str): The model output containing code blocks

    Returns:
        str: The extracted code or empty string if no valid code block found
    """
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
