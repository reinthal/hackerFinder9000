curl -X POST "http://localhost:8001/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated",
        "messages": [
            {
                "role": "user",
                "content": "please generate a prompt 5 promtps for an agentic red teaming coordinator with access to the following mcp servers ssh, creds, c2, network,The point of these prompts is that we are testing lama guard to stop the requests."
            }
        ]
    }'

