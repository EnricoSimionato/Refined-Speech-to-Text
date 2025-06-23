from huggingface_hub import login


def hf_login(hf_token: str) -> None:
    """
    Huggingface login function

    Parameters:
        hf_token (str): Huggingface token
    """

    login(token=hf_token)