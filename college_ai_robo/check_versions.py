
with open("versions.txt", "w") as f:
    try:
        import pandas
        f.write(f"pandas=={pandas.__version__}\n")
    except ImportError:
        f.write("pandas (not installed)\n")

    try:
        import gradio
        f.write(f"gradio=={gradio.__version__}\n")
    except ImportError:
        f.write("gradio (not installed)\n")

    try:
        import transformers
        f.write(f"transformers=={transformers.__version__}\n")
    except ImportError:
        f.write("transformers (not installed)\n")

    try:
        import torch
        f.write(f"torch=={torch.__version__}\n")
    except ImportError:
        f.write("torch (not installed)\n")

    try:
        import openpyxl
        f.write(f"openpyxl=={openpyxl.__version__}\n")
    except ImportError:
        f.write("openpyxl (not installed)\n")

    try:
        import sklearn
        f.write(f"scikit-learn=={sklearn.__version__}\n")
    except ImportError:
        f.write("scikit-learn (not installed)\n")

    try:
        import numpy
        f.write(f"numpy=={numpy.__version__}\n")
    except ImportError:
        f.write("numpy (not installed)\n")

    try:
        import joblib
        f.write(f"joblib=={joblib.__version__}\n")
    except ImportError:
        f.write("joblib (not installed)\n")

    try:
        import sentencepiece
        f.write(f"sentencepiece=={sentencepiece.__version__}\n")
    except ImportError:
        f.write("sentencepiece (not installed)\n")
