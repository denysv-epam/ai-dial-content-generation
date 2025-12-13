import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._models.message import Message
from task._models.role import Role
from task._utils.bucket_client import DialBucketClient
from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT, DIAL_URL
from task._utils.model_client import DialModelClient


# TODO:
#  1. Create DialBucketClient
#  2. Open image file
#  3. Use BytesIO to load bytes of image
#  4. Upload file with client
#  5. Return Attachment object with title (file name), url and type (mime type)
async def _put_image() -> Attachment:
    """upload image to the bucket and return image data"""

    file_name = "dialx-banner.png"
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = "image/png"

    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        with open(image_path, "rb") as file:
            image_bytes = file.read()

        image_content = BytesIO(image_bytes)

        attachment = await bucket_client.put_file(
            name=file_name, mime_type=mime_type_png, content=image_content
        )

        return Attachment(
            title=file_name, url=attachment.get("url"), type=mime_type_png
        )


# TODO:
#  1. Create DialModelClient
#  2. Upload image (use `_put_image` method )
#  3. Print attachment to see result
#  4. Call chat completion via client with list containing one Message:
#    - role: Role.USER
#    - content: "What do you see on this picture?"
#    - custom_content: CustomContent(attachments=[attachment])
#  ---------------------------------------------------------------------------------------------------------------
#  Note: This approach uploads the image to DIAL bucket and references it via attachment. The key benefit of this
#        approach that we can use Models from different vendors (OpenAI, Google, Anthropic). The DIAL Core
#        adapts this attachment to Message content in appropriate format for Model.
#  TRY THIS APPROACH WITH DIFFERENT MODELS!
#  Optional: Try upload 2+ pictures for analysis
def start() -> None:
    client = DialModelClient(
        deployment_name="gpt-4o",
        api_key=API_KEY,
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
    )

    attachment = asyncio.run(_put_image())
    print("➡️", attachment)

    message = Message(
        role=Role.USER,
        content="Describe the image",
        custom_content=CustomContent(attachments=[attachment]),
    )

    client.get_completion(messages=[message])


start()
