import torch

from flux_pipeline import FluxPipeline
import gradio as gr  # type: ignore
from PIL import Image


def create_demo(
    config_path: str,
):
    generator = FluxPipeline.load_pipeline_from_config_path(config_path)

    def generate_image(
        prompt,
        width,
        height,
        num_steps,
        guidance,
        seed,
        init_image,
        image2image_strength,
        add_sampling_metadata,
    ):

        seed = int(seed)
        if seed == -1:
            seed = None
        out = generator.generate(
            prompt,
            width,
            height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            init_image=init_image,
            strength=image2image_strength,
            silent=False,
            num_images=1,
            return_seed=True,
        )
        image_bytes = out[0]
        return Image.open(image_bytes), str(out[1]), None

    is_schnell = generator.config.version == "flux-schnell"

    with gr.Blocks() as demo:
        gr.Markdown(f"# Flux Image Generation Demo - Model: {generator.config.version}")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value='a photo of a forest with mist swirling around the tree trunks. The word "FLUX" is painted over it in big, red brush strokes with visible texture',
                )
                do_img2img = gr.Checkbox(
                    label="Image to Image", value=False, interactive=not is_schnell
                )
                init_image = gr.Image(label="Input Image", visible=False)
                image2image_strength = gr.Slider(
                    0.0, 1.0, 0.8, step=0.1, label="Noising strength", visible=False
                )

                with gr.Accordion("Advanced Options", open=False):
                    width = gr.Slider(128, 8192, 1152, step=16, label="Width")
                    height = gr.Slider(128, 8192, 640, step=16, label="Height")
                    num_steps = gr.Slider(
                        1, 50, 4 if is_schnell else 20, step=1, label="Number of steps"
                    )
                    guidance = gr.Slider(
                        1.0,
                        10.0,
                        3.5,
                        step=0.1,
                        label="Guidance",
                        interactive=not is_schnell,
                    )
                    seed = gr.Textbox(-1, label="Seed (-1 for random)")
                    add_sampling_metadata = gr.Checkbox(
                        label="Add sampling parameters to metadata?", value=True
                    )

                generate_btn = gr.Button("Generate")

            with gr.Column(min_width="960px"):
                output_image = gr.Image(label="Generated Image")
                seed_output = gr.Number(label="Used Seed")
                warning_text = gr.Textbox(label="Warning", visible=False)
                # download_btn = gr.File(label="Download full-resolution")

        def update_img2img(do_img2img):
            return {
                init_image: gr.update(visible=do_img2img),
                image2image_strength: gr.update(visible=do_img2img),
            }

        do_img2img.change(
            update_img2img, do_img2img, [init_image, image2image_strength]
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                width,
                height,
                num_steps,
                guidance,
                seed,
                init_image,
                image2image_strength,
                add_sampling_metadata,
            ],
            outputs=[output_image, seed_output, warning_text],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument(
        "--config", type=str, default="configs/config-dev.json", help="Config file path"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public link to your demo"
    )
    args = parser.parse_args()

    demo = create_demo(args.config)
    demo.launch(share=args.share)
