
import torch
import torch.onnx



if __name__ == '__main__':
    import sys
    from model import load_frozen_model
    model = load_frozen_model(sys.argv[1])

    input_tensor = torch.zeros((6, 7), dtype=torch.int8)

    torch.onnx.export(
        model,                  # model to export
        (input_tensor,),        # inputs of the model,
        "export.onnx",          # filename of the ONNX model
        input_names=["board"],  # Rename inputs for the ONNX model
        output_names=["logits", "value"],  # Rename outputs for the ONNX model
        dynamo=False,             # True or False to select the exporter to use
        use_external_data_format=False,
        #verbose=True,
        #verify=True
    )
