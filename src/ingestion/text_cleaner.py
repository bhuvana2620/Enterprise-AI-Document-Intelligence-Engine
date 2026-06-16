import re


def clean_text(text):
    """
    Clean extracted document text.
    """

    # ---------------------------------------------------
    # Remove extra whitespace
    # Example:
    # "Hello     world" -> "Hello world"
    # ---------------------------------------------------
    text = re.sub(r"\s+", " ", text)

    # ---------------------------------------------------
    # Remove repeated newlines
    # Example:
    # "\n\n\n" -> "\n"
    # ---------------------------------------------------
    text = re.sub(r"\n+", "\n", text)

    # ---------------------------------------------------
    # Remove tabs
    # ---------------------------------------------------
    text = text.replace("\t", " ")

    # ---------------------------------------------------
    # Remove weird unicode characters
    # ---------------------------------------------------
    text = text.replace("\xa0", " ")

    # ---------------------------------------------------
    # Strip leading/trailing whitespace
    # ---------------------------------------------------
    text = text.strip()

    return text


# ---------------------------------------------------
# Test Runner
# ---------------------------------------------------
if __name__ == "__main__":

    sample_text = """
    Hello     world!


    This   is   messy text. \t \t


    AI    pipelines are important.
    """

    cleaned = clean_text(sample_text)

    print("===== CLEANED TEXT =====\n")

    print(cleaned)