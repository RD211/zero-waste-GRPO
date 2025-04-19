import requests
from typing import List
from vllm import RequestOutput

def generate(
    problems: List[str],
    meta: dict = {},
    server_port: int = 8005,
) -> List[RequestOutput]:
    server_url = f"http://localhost:{server_port}/generate"


    response = requests.post(server_url, json={"problems": problems, "meta": meta})
    response.raise_for_status() 

    response_list = response.json()["generations"]
    results: list[str] = [item for item in response_list]
    return results
