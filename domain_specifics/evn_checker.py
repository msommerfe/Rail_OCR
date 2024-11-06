import json

class EVNChecker:

    def __init__(self, railwaysdatafile):
        with open(railwaysdatafile, 'r', encoding='utf-8') as f:
            self.railway_data_from_json = json.load(f)

    def is_country_code_valid(self, country_code):
        if len(str(country_code)) != 2:
            print(f"CountryCode {country_code} must have two digets")
            return False

        for data in self.railway_data_from_json.values():
            if str(data['Zahlen-Code']) == str(country_code):
                return True
            # Überprüfen, ob der Zahlen-Code eine Liste ist (wie bei Bosnien-Herzegowina)
            if isinstance(data['Zahlen-Code'], list) and str(country_code) in str(data['Zahlen-Code']):
                return True
        return False



    def calculate_EVN_check_digit(self, number):
        """
        Berechnet die Prüfziffer für eine gegebene Grundnummer. Die Grundnummer muss aus 11 Ziffern bestehen.

        Parameters:
        - number: Grundnummer als String

        Returns:
        - Prüfziffer als Integer
        """
        if len(number) != 11:
            return None

        total_sum = 0
        length = len(number)

        for index in range(length):
            digit = int(number[length - 1 - index])
            if index % 2 == 0:  # Ungerade Stellen (von rechts gesehen)
                product = digit * 2
                total_sum += product if product < 10 else product - 9  # Summe der Ziffern des Produkts
            else:  # Gerade Stellen (von rechts gesehen)
                total_sum += digit

        check_digit = (10 - (total_sum % 10)) % 10
        return check_digit

    def is_valid_EVN(self, EVN, debug = True):
        if len(str(EVN)) != 12:
            if debug:
                print(f"EVN {EVN} must have 12 digets")
            return False

        if not self.is_country_code_valid(EVN[2:4]):
            if debug:
                print(f"EVN {EVN} has no valid countrycode")
            return False

        pruefziffer = self.calculate_EVN_check_digit(EVN[0:11])
        if str(pruefziffer) != str(EVN[-1]):
            if debug:
                print(f"EVN {EVN} has invalid pruefziffer")
            return False

        return True
