from ttictoc import Timer

# Define tic and toc functions
_TICTOC_HELPER_CLASS_5da0381c_27af_4d67_8881 = Timer(matlab_like=True)
tic = _TICTOC_HELPER_CLASS_5da0381c_27af_4d67_8881.start
def toc():
    print(f"elapsed: {_TICTOC_HELPER_CLASS_5da0381c_27af_4d67_8881.stop()}s", flush=True)
