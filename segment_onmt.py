#!python3
import argparse
import logging
import tensorflow as tf
import opennmt as onmt

# {GPT2Small,ListenAttendSpell,LstmCnnCrfTagger,LuongAttention,NMTBigV1,NMTMediumV1,NMTSmallV1,ScalingNmtEnDe,ScalingNmtEnFr,Transformer,TransformerBase,TransformerBaseRelative,TransformerBaseSharedEmbeddings,TransformerBig,TransformerBigRelative,TransformerBigSharedEmbeddings,TransformerRelative,TransformerTiny}

tf.get_logger().setLevel(logging.INFO)
# os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION","python")

class CustomLstmCnnCrfTagger(onmt.models.sequence_tagger.SequenceTagger):
    """Defines a bidirectional LSTM-CNNs-CRF as described in https://arxiv.org/abs/1603.01354."""

    def __init__(self):
        super().__init__(
            #inputter=inputters.MixedInputter(
            #    [
            #        inputters.WordEmbedder(embedding_size=64),
            inputter = onmt.inputters.CharConvEmbedder(
                        embedding_size=64,
                        num_outputs=30,
                        kernel_size=5,
                        stride=1,
                        dropout=0.2,
                    ),
             #   ],
             #   dropout=0.2,
            #),
            encoder = onmt.encoders.RNNEncoder(
                num_layers=2,
                num_units=256,
                bidirectional=True,
                dropout=0.1,
                residual_connections=True,
                cell_class=tf.keras.layers.LSTMCell,
            ),
            crf_decoding=True,
        )

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("run", choices=["train", "translate"], help="Run type.")
    parser.add_argument("--src", required=True, help="Path to the source file.")
    parser.add_argument("--tgt", help="Path to the target file.")
    parser.add_argument(
        "--src_vocab", required=True, help="Path to the source vocabulary."
    )
    parser.add_argument(
        "--tgt_vocab", required=True, help="Path to the target vocabulary."
    )
    parser.add_argument("--src_val")
    parser.add_argument("--tgt_val")
    parser.add_argument(
        "--model_dir",
        default="checkpoint",
        help="Directory where checkpoint are written.",
    )
    args = parser.parse_args()
    print(args)

    config = {
        "model_dir": args.model_dir,
        "data":{
            "source_vocabulary": args.src_vocab,
            "target_vocabulary": args.tgt_vocab,
            "eval_features_file": args.src_val,
            "eval_labels_file": args.tgt_val,
            "train_features_file": args.src,
            "train_labels_file": args.tgt,

           },
        "train":{
            "batch_size": 32,
            "max_step": 1000000,
            "save_checkpoints_steps":500
        },
        "params":{
            "optimizer":"Adam",
            "learning_rate":0.0002
        },
        "eval":{
            "steps":500,
            "export_on_best":"accuracy",
            "export_format":"checkpoint",
            "early_stopping":{
                "metric":"accuracy",
                "min_improvement":0.001,
                "steps":4,
                }
        }
    }
   
    model = onmt.models.LstmCnnCrfTagger()
    runner = onmt.Runner(model, config)

    if args.run == "train":
        runner.train(with_eval=True)
    elif args.run == "translate":
        runner.infer(args.src)

if __name__ == "__main__":
    main()
