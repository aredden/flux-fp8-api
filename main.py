import argparse
import uvicorn
from api import app


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Flux API server")
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8088,
        help="Port to run the server on",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on",
    )
    parser.add_argument(
        "-f", "--flow-model-path", type=str, help="Path to the flow model"
    )
    parser.add_argument(
        "-t", "--text-enc-path", type=str, help="Path to the text encoder"
    )
    parser.add_argument(
        "-a", "--autoencoder-path", type=str, help="Path to the autoencoder"
    )
    parser.add_argument(
        "-m",
        "--model-version",
        type=str,
        choices=["flux-dev", "flux-schnell"],
        default="flux-dev",
        help="Choose model version",
    )
    parser.add_argument(
        "-F",
        "--flux-device",
        type=str,
        default="cuda:0",
        help="Device to run the flow model on",
    )
    parser.add_argument(
        "-T",
        "--text-enc-device",
        type=str,
        default="cuda:0",
        help="Device to run the text encoder on",
    )
    parser.add_argument(
        "-A",
        "--autoencoder-device",
        type=str,
        default="cuda:0",
        help="Device to run the autoencoder on",
    )
    parser.add_argument(
        "-q",
        "--num-to-quant",
        type=int,
        default=20,
        help="Number of linear layers in flow transformer (the 'unet') to quantize",
    )
    parser.add_argument(
        "-C",
        "--compile",
        action="store_true",
        default=False,
        help="Compile the flow model with extra optimizations",
    )
    parser.add_argument(
        "-qT",
        "--quant-text-enc",
        type=str,
        default="qfloat8",
        choices=["qint4", "qfloat8", "qint2", "qint8", "bf16"],
        help="Quantize the t5 text encoder to the given dtype, if bf16, will not quantize",
        dest="quant_text_enc",
    )
    parser.add_argument(
        "-qA",
        "--quant-ae",
        action="store_true",
        default=False,
        help="Quantize the autoencoder with float8 linear layers, otherwise will use bfloat16",
        dest="quant_ae",
    )
    parser.add_argument(
        "-OF",
        "--offload-flow",
        action="store_true",
        default=False,
        dest="offload_flow",
        help="Offload the flow model to the CPU when not being used to save memory",
    )
    parser.add_argument(
        "-OA",
        "--no-offload-ae",
        action="store_false",
        default=True,
        dest="offload_ae",
        help="Disable offloading the autoencoder to the CPU when not being used to increase e2e inference speed",
    )
    parser.add_argument(
        "-OT",
        "--no-offload-text-enc",
        action="store_false",
        default=True,
        dest="offload_text_enc",
        help="Disable offloading the text encoder to the CPU when not being used to increase e2e inference speed",
    )
    parser.add_argument(
        "-PF",
        "--prequantized-flow",
        action="store_true",
        default=False,
        dest="prequantized_flow",
        help="Load the flow model from a prequantized checkpoint "
        + "(requires loading the flow model, running a minimum of 24 steps, "
        + "and then saving the state_dict as a safetensors file), "
        + "which reduces the size of the checkpoint by about 50% & reduces startup time",
    )
    parser.add_argument(
        "-nqfm",
        "--no-quantize-flow-modulation",
        action="store_false",
        default=True,
        dest="quantize_modulation",
        help="Disable quantization of the modulation layers in the flow model, adds ~2GB vram usage for moderate precision improvements",
    )
    parser.add_argument(
        "-qfl",
        "--quantize-flow-embedder-layers",
        action="store_true",
        default=False,
        dest="quantize_flow_embedder_layers",
        help="Quantize the flow embedder layers in the flow model, saves ~512MB vram usage, but precision loss is very noticeable",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # lazy loading so cli returns fast instead of waiting for torch to load modules
    from flux_pipeline import FluxPipeline
    from util import load_config, ModelVersion

    if args.config_path:
        app.state.model = FluxPipeline.load_pipeline_from_config_path(
            args.config_path, flow_model_path=args.flow_model_path
        )
    else:
        model_version = (
            ModelVersion.flux_dev
            if args.model_version == "flux-dev"
            else ModelVersion.flux_schnell
        )
        config = load_config(
            model_version,
            flux_path=args.flow_model_path,
            flux_device=args.flux_device,
            ae_path=args.autoencoder_path,
            ae_device=args.autoencoder_device,
            text_enc_path=args.text_enc_path,
            text_enc_device=args.text_enc_device,
            flow_dtype="float16",
            text_enc_dtype="bfloat16",
            ae_dtype="bfloat16",
            num_to_quant=args.num_to_quant,
            compile_extras=args.compile,
            compile_blocks=args.compile,
            quant_text_enc=(
                None if args.quant_text_enc == "bf16" else args.quant_text_enc
            ),
            quant_ae=args.quant_ae,
            offload_flow=args.offload_flow,
            offload_ae=args.offload_ae,
            offload_text_enc=args.offload_text_enc,
            prequantized_flow=args.prequantized_flow,
            quantize_modulation=args.quantize_modulation,
            quantize_flow_embedder_layers=args.quantize_flow_embedder_layers,
        )
        app.state.model = FluxPipeline.load_pipeline_from_config(config)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
