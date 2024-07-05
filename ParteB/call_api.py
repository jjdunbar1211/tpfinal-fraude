import requests

search_api_url = 'http://0.0.0.0:7860/prediccion'


'''
# CASO 1 -> Tipo de fraude: 0/False
data = {
    "orderAmount" : 18.0,
    "orderState" : "pending",
    "paymentMethodRegistrationFailure" : "True",
    "paymentMethodType" : "card",
    "paymentMethodProvider" : "JCB 16 digit",
    "paymentMethodIssuer" : "Citizens First Banks",
    "transactionAmount" : 18,
    "transactionFailed" : "False",
    "emailDomain" : "com",
    "emailProvider" : "yahoo",
    "customerIPAddressSimplified" : "only_letters",
    "sameCity" : "yes"
    }

'''


# CASO 2 -> Tipo de fraude: 1/True
data = {
    "orderAmount" : 26.0,
    "orderState" : "fulfilled",
    "paymentMethodRegistrationFailure" : "True",
    "paymentMethodType" : "bitcoin",
    "paymentMethodProvider" : "VISA 16 digit",
    "paymentMethodIssuer" : "Solace Banks",
    "transactionAmount" : 26,
    "transactionFailed" : "False",
    "emailDomain" : "com",
    "emailProvider" : "yahoo",
    "customerIPAddressSimplified" : "only_letters",
    "sameCity" : "no"
}




response = requests.post(search_api_url, json=data)
print(response.json())

#response = requests.get('http://0.0.0.0:7860/prediccion')
#print(response.json())

#response = requests.get('http://0.0.0.0:7860/')
#print(response.json())

#response = await requests.get('http://127.0.0.1:8000/edvai')
#print(response.json())