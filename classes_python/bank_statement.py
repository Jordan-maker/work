
class bank_account:

    CountryAndCurrency = {'mexico': 'MXN',
                          'usa': 'USD',
                          'colombia': 'COP'}

    def __init__(self, bank_name, user_name:str='User', country:str='usa'):
        self.bank_name = bank_name
        self.user_name = user_name
        self.country = country.lower()
        self.currency = self.CountryAndCurrency[self.country]

        self.balance = 0
        self.shoping = list()

    def welcome_message(self):
        print(f"Hi {self.user_name}, welcome to {self.bank_name}!\n")

    def show_balance(self):
        print(f"Your balance is {self.balance} {self.currency}.\n")

    def show_shopping(self):
        print(f"Next, purchases are displayed:")
        for purchase in self.shoping:
            print(purchase)

    def deposit_money(self, amount):
        print(f"{amount} {self.currency} was deposited to your bank account.\n")
        self.balance += amount

    def check_money(self, amount, movement=True):
        if amount > self.balance:
            print("Sorry, not enought money in your balance. Operation denied.\n")
            movement = False
        return movement

    def ask_confirmation(self, answer=''):
        answer = input("Are you sure for making this operation (Y/N):").lower()
        while answer not in ['y', 'yes'] + ['n', 'no']:
            print("Given value not allowed. Please, replay your answer.")
            answer = input("Are you sure for making this operation (Y/N):").lower()
        if answer in ['y', 'yes']: return True
        elif answer in ['n', 'no']: return False

    def withdraw_money(self, amount):
        movement = bank_account.check_money(self, amount)
        if movement:
            self.balance -= amount
            print(f"{amount} {self.currency} was withdrawed from your bank account.\n")

    def buy(self, amount, date, product='', store=''):
        movement = bank_account.check_money(self, amount)
        if movement:
            print(f"You are close to buy -{product}- for {amount} {self.currency}.")
            answer = bank_account.ask_confirmation(self)
            if answer:
                self.balance -= amount
                self.shoping.append({'amount': amount,
                                     'date': date,
                                     'product': product,
                                     'store':store
                                     })
                if product and store:
                    print(f"The product -{product}- was bought in {store} for {amount} {self.currency} on {date}.\n")
                elif product and not store:
                    print(f"The product -{product}- was bought for {amount} {self.currency} on {date}.\n")
                else:
                    print(f"The product was bought for {amount} {self.currency} on {date}.\n")
            else:
                print("Operation canceled.\n")


def main():
    my_account = bank_account(bank_name="BBVA", user_name='Jordan', country='Mexico')
    my_account.welcome_message()
    my_account.deposit_money(amount=1000)
    my_account.buy(amount=400, date='01/05/2023', product='shoes', store='Adidas')
    my_account.buy(amount=200, date='01/05/2023', product='groceries', store='Walmart')
    my_account.withdraw_money(amount=300)
    my_account.show_balance()

    # shopping:
    my_account.show_shopping()

if __name__ == '__main__':
    main()
