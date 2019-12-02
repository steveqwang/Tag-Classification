import requests
import re
import time


header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}


def process_request(url):
    time.sleep(1)

    try:
        response = requests.get(url, headers=header)

        if response.status_code == 200:
            return response.text

    except requests.RequestException:

        return None


if __name__ == '__main__':
    with open('gcp_title.txt', 'w') as f:

        for i in range(1, 17914 // 50):

            print(str(i) + "   " + str(17914 // 50))

            # url = "https://stackoverflow.com/questions/tagged/azure?tab=newest&page=" + str(i) + "&pagesize=50"
            # url = "https://stackoverflow.com/questions/tagged/amazon-web-services?tab=newest&page=" + str(i) + "&pagesize=50"
            url_ = "https://stackoverflow.com/questions/tagged/google-cloud-platform?tab=newest&page=" + str(i) + "&pagesize=50"

            html = process_request(url_)
            pattern = re.compile('<a href=.*? class="question-hyperlink">.*?</a>')

            try:
                items = re.findall(pattern, html)

                for item in items:
                    f.write(str(item[item.index('>') + 1:item.index('</')]) + '\n')
            except:
                print("Lost: " + url_)
