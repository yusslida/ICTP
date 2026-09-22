from qkeras import QDense, QActivation

def quantized_model(bestHP):

    model_q = keras.Sequential([
        keras.Input(shape=(61,), name='inputLayer'),

        QDense(
            bestHP[0],
            kernel_quantizer=kernelQ,
            bias_quantizer=biasQ,
            name='fc1'
        ),
        QActivation(activationQ, name='relu1'),

        QDense(
            bestHP[1],
            kernel_quantizer=kernelQ,
            bias_quantizer=biasQ,
            name='fc2'
        ),
        QActivation(activationQ, name='relu2'),

        QDense(
            bestHP[2],
            kernel_quantizer=kernelQ,
            bias_quantizer=biasQ,
            name='fc3'
        ),
        QActivation(activationQ, name='relu3'),

        Dropout(0.1),

        QDense(
            bestHP[3],
            kernel_quantizer=kernelQ,
            bias_quantizer=biasQ,
            name='fc4'
        ),
        QActivation(activationQ, name='relu4'),

        Dropout(0.2),

        QDense(
            bestHP[4],
            kernel_quantizer=kernelQ,
            bias_quantizer=biasQ,
            name='fc5'
        ),
        QActivation(activationQ, name='relu5'),

        Dropout(0.2),

        QDense(
            2,
            kernel_quantizer=kernelQ,
            bias_quantizer=biasQ,
            name='output'
        ),

        Activation('sigmoid', name='outputActivation')
    ])

    return model_q
