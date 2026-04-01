"""Answer extraction utilities.

Ported from TokenSkip (https://github.com/hemingkx/TokenSkip) with additions
for GSM8K #### format and numeric comparison.
"""

import re
import regex


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
        string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("x\\in", "")

    string = string.replace("\\%", "%")
    string = string.replace("\%", "%")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    string = string.replace("\\cdot", "")

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")
    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace("\"", "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string


def extract_boxed_answers(text):
    """Extract all \\boxed{...} answers from text, handling nested braces."""
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_answer(pred_str, exhaust=False):
    """Extract answer from model output. Tries boxed, #### separator, 
    'the answer is' pattern, and falls back to last number."""
    pred = []
    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = [tmp.split('$. I hope', 1)[0].strip()]
    elif 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif '####' in pred_str:
        ans = pred_str.split('####')[-1].strip()
        ans = ans.replace(",", "").replace("$", "").replace("%", "")
        if ans:
            pred = [ans]
    elif 'he answer is' in pred_str:
        pred = [pred_str.split('he answer is')[-1].strip()]
    else:
        pattern = r'-?\d*\.?\d+'
        ans = re.findall(pattern, pred_str.replace(",", ""))
        if len(ans) >= 1:
            ans = ans[-1]
        else:
            ans = ''
        if ans:
            pred.append(ans)

    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def extract_gsm8k_answer(response):
    """Extract numeric answer from GSM8K-style response (#### separator or last number)."""
    if '####' in response:
        ans = response.split('####')[-1].strip()
        ans = ans.replace(",", "").replace("$", "").replace("%", "")
        return ans

    pattern = r'-?\d*\.?\d+'
    numbers = re.findall(pattern, response.replace(",", ""))
    if numbers:
        return numbers[-1]
    return ""


def extract_math_answer(response):
    """Extract answer from MATH-style response (\\boxed{} or fallback)."""
    return extract_answer(response, exhaust=False)


def answers_equal(predicted, expected, prec=1e-3):
    """Compare two answers: try numeric comparison first, then string match."""
    if not predicted or not expected:
        return False

    predicted = str(predicted).strip()
    expected = str(expected).strip()

    if predicted == expected:
        return True

    predicted_norm = strip_string(predicted)
    expected_norm = strip_string(expected)
    if predicted_norm == expected_norm:
        return True

    try:
        pred_f = float(regex.sub(r',', '', predicted))
        exp_f = float(regex.sub(r',', '', expected))
        if abs(pred_f - exp_f) < prec:
            return True
    except (ValueError, TypeError):
        pass

    return False
