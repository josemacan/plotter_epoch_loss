import re
import matplotlib.pyplot as plot

# PATHS
log_path = "/home/admint/PycharmProjects/GraficoLogLoss/logs/"

file_name_captioner = "entrenamiento_captioner.txt"
file_path_captioner = log_path + file_name_captioner

file_name_textenc = "entrenamiento_encodertext.txt"
file_path_textenc  = log_path + file_name_textenc

# Abrir y leer log.txt

def filtro_expreg_loss(txt_path):

    #totalEpochs = 60  # cantidad total de epochs
    numEpoch = []  # lista de epochs
    valLoss = []  # lista de loss

    f = open(txt_path, "r")
    linea = f.readlines()

    indice = 0
    for renglon in linea:
        #print("Linea %d: %s" %(indice,renglon))
        inicio = re.search(r"Epoch 1 Batch 0 Loss\s(\d+.\d+)",renglon)
        if inicio is not None:
            numEpoch.append(0)
            valLoss.append(float(inicio.group(1)))

        resultado = re.search(r"Epoch\s(\d+)\sLoss\s(\d+.\d+)", renglon)
        if resultado is not None:
            numEpoch.append( int(resultado.group(1)) )
            valLoss.append( float(resultado.group(2)) )

        ## Imprimir lista dual

    print("\nElement: %s \n" %txt_path)
    for i in range(len(numEpoch)):
        print("numEpoch: %d -- Loss: %f" % (numEpoch[i], valLoss[i]))

    return numEpoch,valLoss


def plotter(epoch, loss, title,png_name):
    ## Generar grafico y mostrar
    plot.plot(epoch, loss)
    plot.xlabel("Epoch")
    plot.ylabel("Loss")
    plot.title(title)

    figure = plot.gcf()
    plot.show()
    plot.draw()
    figure.savefig(png_name)

    return

captionerEpoch, captionerLoss = filtro_expreg_loss(file_path_captioner)
textencEpoch, textencLoss = filtro_expreg_loss(file_path_textenc)

plotter(captionerEpoch,captionerLoss,"Entrenamiento modelo imagenes", "image_trainloss.png")
plotter(textencEpoch,textencLoss,"Entrenamiento modelo texto", "text_trainloss.png")








