import gradio as gr
import pandas as pd
import pickle
import os


# Define params names
PARAMS_NAME = [
    "orderAmount",
    "orderState",
    "paymentMethodRegistrationFailure",
    "paymentMethodType",
    "paymentMethodProvider",
    "paymentMethodIssuer",
    "transactionAmount",
    "transactionFailed",
    "emailDomain",
    "emailProvider",
    "customerIPAddressSimplified",
    "sameCity"
]

           
# Load model
with open("models/modelo_proyecto_final.pkl", "rb") as f:
    model = pickle.load(f)

# Columnas
COLUMNS_PATH = "models/categories_ohe_without_fraudulent.pickle"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)

MAIN_FOLDER = "models"

BINS_ORDER = os.path.join(MAIN_FOLDER, "saved_bins_order.pickle")
with open(BINS_ORDER, 'rb') as handle:
    saved_bins_order = pickle.load(handle)

BINS_TRANSACTION = os.path.join(MAIN_FOLDER, "saved_bins_transaction.pickle")
with open(BINS_TRANSACTION, 'rb') as handle:
    saved_bins_transaction = pickle.load(handle)


def predict(*args):
    answer_dict = {}

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    # Crear dataframe
    single_instance = pd.DataFrame.from_dict(answer_dict)


    # Manejar puntos de corte o bins
    single_instance["orderAmount"] = single_instance["orderAmount"].astype(float)
    single_instance["orderAmount"] = pd.cut(single_instance['orderAmount'],
                                     bins=saved_bins_order, 
                                     include_lowest=True)
    
    single_instance["transactionAmount"] = single_instance["transactionAmount"].astype(int)
    single_instance["transactionAmount"] = pd.cut(single_instance['transactionAmount'],
                                     bins=saved_bins_order, 
                                     include_lowest=True)
    

    # Reformat columns
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)
    
    prediction = model.predict(single_instance_ohe)

    # Como sabemos el model nos devuelve los tipos de fraude 1, 2 y 3 en el response. Podemos devolver un response estilo semáforo.
    
    # Cast numpy.int64 to just a int
    type_of_fraud = int(prediction[0])


    # Adaptación respuesta
    response = "Error parsing value"
    if type_of_fraud == 0:
        response = "False"
    if type_of_fraud == 1:
        response = "True"
    if type_of_fraud == 2:
        response = "Warning"

    return response


with gr.Blocks() as demo:
    gr.Markdown(
        """
        #   Prevención de Fraude 🕵️‍♂️👀🔍🕵️‍♀️
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## Predecir si un cliente es fraudulento o no.
                """
            )
            
                      
            orderAmount = gr.Slider(
                label="Order amount", 
                minimum=10, 
                maximum=75, 
                step=1, 
                randomize=True
                )

            orderState = gr.Radio(
                label="Order state",
                choices=["failed", "fulfilled", "pending"],
                value="failed"
                )
            
            paymentMethodRegistrationFailure = gr.Radio(
                label="Payment method registration failure",
                choices=["True", "False"],
                value="True"
                )
            
            paymentMethodType = gr.Radio(
                label="Payment Method Type",
                choices=["apple pay", "bitcoin", "card", "paypal"],
                value="bitcoin"
                )

           
            paymentMethodProvider = gr.Dropdown(
                label="Payment method provider",
                choices=["American Express", "Diners Club / Carte Blanche", "Provider_Discover", "Provider_JCB 15 digit", "JCB 16 digit", "Maestro", "Mastercard", "VISA 13 digit", "VISA 16 digit", "Voyager"],
                multiselect=False,
                value="American Express",
                )
            
            paymentMethodIssuer = gr.Dropdown(
                label="Payment method issuer",
                choices=["Bastion Banks", "Bulwark Trust Corp.", "Citizens First Banks", "Fountain Financial Inc.", "Grand Credit Corporation", "Her Majesty Trust", "His Majesty Bank Corp.", "Rose Bancshares", "Solace Banks", "Vertex Bancorp", "weird"],
                multiselect=False,
                value="Bastion Banks",
                )

            transactionAmount = gr.Slider(
                label="Transaction amount", 
                minimum=1, 
                maximum=100, 
                step=1, 
                randomize=True
                )
            
            transactionFailed = gr.Radio(
                label="Transaction failed",
                choices=["True", "False"],
                value="False"
                )
            
            emailDomain = gr.Radio(
                label="Email domain",
                choices=["biz", "com", "info", "net", "org", "weird"],
                value="com"
                )
            
            emailProvider = gr.Radio(
                label="Email provider",
                choices=["gmail", "hotmail", "other", "weird", "yahoo"],
                value="gmail"
                )
            
            customerIPAddressSimplified = gr.Radio(
                label="Customer IP Address",
                choices=["digits_and_letters", "only_letters"],
                value="only_letters"
                )
            
            sameCity = gr.Radio(
                label="Same city",
                choices=["no", "yes", "unknown"],
                value="unknown"
                )
            




        with gr.Column():

            gr.Markdown(
                """
                ## Predicción
                """
            )

            label = gr.Label(label="Tipo de Fraude")
            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
                predict,
                inputs=[
                   orderAmount,
                   orderState,
                   paymentMethodRegistrationFailure,
                   paymentMethodType,
                   paymentMethodProvider,
                   paymentMethodIssuer,
                   transactionAmount,
                   transactionFailed,
                   emailDomain,
                   emailProvider,
                   customerIPAddressSimplified,
                   sameCity,
                ],
                outputs=[label],
                api_name="prediccion"
            )
    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='mailto:csamignone@gmail.com'>
                TP Final Boootcamp DS&MLOps EDVai 🤗
            </a>
        </p>
        """
    )

demo.launch()
