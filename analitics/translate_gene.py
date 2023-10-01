import json
import math
import pywt

def get_channel_from_value(value):
    colors = ['r', 'g', 'b', 'h', 's', 'v', 'y']
    index = math.floor(value * len(colors))
    return colors[index]

def get_wavelet_function_from_value(value):
    waves = pywt.wavelist(kind="discrete")
    index = math.floor(value * len(waves))
    return waves[index]

def translate_color(value):
    color = get_channel_from_value(value)
    if color == "r":
        return "Vermelho (RGB)"
    elif color == "g":
        return "Verde (RGB)"
    elif color == "b":
        return "Azul (RGB)"
    elif color == "h":
        return "Matiz (HSV)"
    elif color == "s":
        return "Saturação (HSV)"
    elif color == "v":
        return "Valor (HSV)"
    else:
        return "Escala de Cinza"

def translate_boolean(v):
    if v>0.5:
        return "Sim"
    else:
        return "Não"

def map_wavelet_repeats(value: float):
    return math.floor(value * 30) + 1

def map_sigma(value: float):
    return math.floor(value * 16) * 2 + 1  # 1, 3, 5,...,31

def wavelet_full_name(wav):
    wavelet = get_wavelet_function_from_value(wav)
    return f"{pywt.Wavelet(wavelet).family_name} ({pywt.Wavelet(wavelet).name})"

def get_properties(di):
    prop = {}
    prop[0] = translate_color(di["COLOR"])
    prop[1] = translate_boolean(di["HISTOGRAM"])
    prop[2] = translate_boolean(di["DOG"])
    prop[3] = map_sigma(di["SIGMA1"])
    prop[4] = map_sigma(di["SIGMA2"])
    prop[5] = wavelet_full_name(di["WAVELET"])
    prop[6] = map_wavelet_repeats(di["WAVELET_REPEATS"])
    prop[7] = translate_boolean(di["HORIZONTAL"])
    prop[8] = translate_boolean(di["VERTICAL"])
    prop[9] = translate_boolean(di["DIAGONAL"])
    prop[10] = translate_boolean(di["APPROXIMATION"])
    prop[11] = translate_boolean(di["MEAN"])
    prop[12] = translate_boolean(di["MEDIAN"])
    prop[13] = translate_boolean(di["VARIANCE"])
    prop[14] = translate_boolean(di["ENERGY"])
    return prop


def apply_in_template(title, classifier, path):
    data = []
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            value = f"{line.split('}: ')[0]} }}".replace("\'", "\"")
            dictData = json.loads(value)
            data.append(dictData)

    prop_0 = get_properties(data[0])
    prop_20 = get_properties(data[-1])

    template = f'''
\\begin{{table}}[]
  \\centering
  \\begin{{tabular}}{{ l|l l }}
   
    \\textbf{{{title} usando {classifier}}} & \\textbf{{Geração 01}} & \\textbf{{Geração 20}} \\\\ \\hline
    \\textbf{{Canal de Cor}}               & {prop_0[0]} & {prop_20[0]}   \\\\ 
    \\textbf{{Equalização de Histograma}}  & {prop_0[1]} & {prop_20[1]}   \\\\ 
    \\textbf{{Diferença de Gaussianas}}    & {prop_0[2]} & {prop_20[2]}   \\\\ 
    \\textbf{{Sigma 1}}                    & {prop_0[3]}  & {prop_20[3]}  \\\\ 
    \\textbf{{Sigma 2}}                    & {prop_0[4]}  & {prop_20[4]}  \\\\ 
    \\textbf{{Função Wavelet Mãe}}         & {prop_0[5]}  & {prop_20[5]}  \\\\ 
    \\textbf{{Repetições Wavelet}}         & {prop_0[6]}  & {prop_20[6]}  \\\\ 
    \\textbf{{Detalhes Horizontais}}       & {prop_0[7]}  & {prop_20[7]}  \\\\ 
    \\textbf{{Detalhes Verticais}}         & {prop_0[8]}  & {prop_20[8]}  \\\\ 
    \\textbf{{Detalhes Diagonais}}         & {prop_0[9]}  & {prop_20[9]}  \\\\ 
    \\textbf{{Aproximação}}                & {prop_0[10]} & {prop_20[10]} \\\\ 
    \\textbf{{Energia}}                    & {prop_0[11]} & {prop_20[11]} \\\\ 
    \\textbf{{Média}}                      & {prop_0[12]} & {prop_20[12]} \\\\ 
    \\textbf{{Mediana}}                    & {prop_0[13]} & {prop_20[13]} \\\\ 
    \\textbf{{Variância}}                  & {prop_0[14]} & {prop_20[14]} \\\\ 
  \\end{{tabular}}
\\end{{table}}
    '''

    # print(template)
    with open(f"{title}_{classifier}.tex","w") as file:
        file.write(template)

if __name__ == "__main__":
    experiments = [
        ("LLC x LF", "FL_CLL"),
        ("LLC x LCM", "CLL_MCL"),
        ("LF x LCM", "FL_MCL" )
    ]




    for (title, fold) in experiments:
        classifiers =[
            ("Random Forest","tab:blue", f"../results/{fold}_RANDOM_FOREST/best.txt"),
            ("Ada Boost","tab:orange", f"../results/{fold}_ADA_BOOST/best.txt"),
            ("SVM", "tab:green", f"../results/{fold}_LINEAR_SVM/best.txt")
        ]

        for (classifier, color, file) in classifiers:
            apply_in_template(title, classifier, file)