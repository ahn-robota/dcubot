import onnxruntime


class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs", awd=False, num_threads=None):
        """num_threads: caps onnxruntime's intra/inter-op thread pools --
        unset (default) leaves onnxruntime's own unconstrained default,
        which on a weak multi-core board (Raspberry Pi Zero 2 W) can spin
        up all cores on every single inference call. hardware/pi/
        policy_runner.py passes 1. Added 2026-11-11 while chasing a
        real-hardware USB-serial dropout that turned out (2026-11-13,
        see docs/HARDWARE_SETUP.md) to be a disconnected ground wire on
        the servo power barrel jack, not inference CPU load -- kept
        anyway since it's harmless and still a reasonable thing to pin
        down on a weak board."""
        self.onnx_model_path = onnx_model_path
        sess_options = onnxruntime.SessionOptions()
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = input_name
        self.awd = awd

    def infer(self, inputs):
        if self.awd:
            outputs = self.ort_session.run(None, {self.input_name: [inputs]})
            return outputs[0][0]
        else:
            outputs = self.ort_session.run(
                None, {self.input_name: inputs.astype("float32")}
            )
            return outputs[0]


if __name__ == "__main__":
    import argparse
    import numpy as np
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    args = parser.parse_args()

    obs_size = 46

    oi = OnnxInfer(args.onnx_model_path, awd=True)
    times = []
    for i in range(1000):
        inputs = np.random.uniform(size=obs_size).astype(np.float32)
        # inputs = np.arange(obs_size).astype(np.float32)
        # print(inputs)
        start = time.time()
        print(oi.infer(inputs))
        times.append(time.time() - start)

    print("Average time: ", sum(times) / len(times))
    print("Average fps: ", 1 / (sum(times) / len(times)))
