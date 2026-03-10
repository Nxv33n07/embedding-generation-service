from openai import OpenAI

# Inference Check for GPT-OSS
client = OpenAI(
    base_url="http://3.109.63.164/gptoss/v1",
    api_key="dummy"
)

print("🚀 testing GPT-OSS inference...")

try:
    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant for an ecommerce platform."},
            {"role": "user", "content": "Hello! Can you help me find some products?"}
        ]
    )

    print("\n✅ Response from GPT-OSS:")
    print("-" * 30)
    print(response.choices[0].message.content)
    print("-" * 30)

except Exception as e:
    print(f"\n❌ Inference failed: {e}")
