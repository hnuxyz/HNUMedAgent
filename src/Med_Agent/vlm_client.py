from openai import OpenAI
import base64
import io

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:23333/v1"
)


def encode_image(image):

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def vlm_infer(image1, image2, text):

    content = []

    content.append({"type":"text","text":text})

    if image1 is not None:
        img1 = encode_image(image1)

        content.append({
            "type":"image_url",
            "image_url":{
                "url":f"data:image/png;base64,{img1}"
            }
        })

    if image2 is not None:
        img2 = encode_image(image2)

        content.append({
            "type":"image_url",
            "image_url":{
                "url":f"data:image/png;base64,{img2}"
            }
        })

    completion = client.chat.completions.create(
        model="intern-s1-mini",
        messages=[
            {
                "role":"user",
                "content":content
            }
        ]
    )

    return completion.choices[0].message.content