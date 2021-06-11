import requests

url = 'My URL goes here'
r = requests.post(url,json={'loan_status': 0, 'loan_amount':1000, 
                            'inq_last_6mths': 1, 'revol_util': 0.8, 
                            'pub_rec_bankruptcies': 1, 'ln_annual_inc': 200,
                            'purpose': 'credit_card'})

print(r.json())

