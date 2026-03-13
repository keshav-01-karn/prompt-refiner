import os
import gradio as gr
from google import genai
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def refine_prompt(user_prompt):
    try:
        instruction = (
            "You are an expert in prompt engineering. Improve the following prompt. "
            "Make it clearer, more specific, and optimized for better AI responses. "
            "Keep the same tone unless instructed otherwise.\n\n"
            f"Original prompt:\n{user_prompt}\n\nRefined prompt:"
        )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=instruction
        )

        return response.text.strip()

    except Exception as e:
        return f"Error refining prompt: {str(e)}"

examples = [
    ["Write about climate change"],
    ["Explain Newton's laws"],
    ["Create a marketing email"],
    ["Tell me about artificial intelligence"]
]

with gr.Blocks(title="Prompt Refiner") as demo:

    gr.Markdown(
        """
        # ✨ Prompt Refiner
        
        Turn a **rough or vague prompt** into a **clear and optimized prompt**  
        designed to produce better responses from AI models.
        """
    )

    with gr.Row():
        input_prompt = gr.Textbox(
            label="Enter your original prompt",
            placeholder="Example: Write about AI",
            lines=3
        )

    refine_button = gr.Button("Refine Prompt 🚀")

    output_prompt = gr.Textbox(
        label="Refined Prompt",
        lines=5
    )

    refine_button.click(
        fn=refine_prompt,
        inputs=input_prompt,
        outputs=output_prompt
    )

    gr.Examples(
        examples=examples,
        inputs=input_prompt
    )

    gr.Markdown(
        """
        Built with **Gradio + Gemini API**
        """
    )

demo.launch(share=True)