import requests
from bs4 import BeautifulSoup
import csv

# 请求URL
url = 'https://movie.douban.com/top250'
# 请求头部
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}


# 解析页面函数
def parse_html(html):
    soup = BeautifulSoup(html, 'lxml')
    movie_list = soup.find('ol', class_='grid_view').find_all('li')
    for movie in movie_list:
        title = movie.find('div', class_='hd').find('span', class_='title').get_text()
        rating_num = movie.find('div', class_='star').find('span', class_='rating_num').get_text()
        comment_num = movie.find('div', class_='star').find_all('span')[-1].get_text()
        introduction = movie.find('div', class_='bd').find('p', class_='').get_text()
        # 以换行符为分隔符，将introduction切为一个列表
        introduction_split = introduction.split('\n')
        if(movie.find('div', class_='bd').find('span', class_='inq')):
            quote = movie.find('div', class_='bd').find('span', class_='inq').get_text()
            writer.writerow([title, rating_num, comment_num, quote, introduction_split[1].strip(), introduction_split[2].strip()])
        else:
            writer.writerow([title, rating_num, comment_num, '', introduction_split[1].strip(), introduction_split[2].strip()])

        '''
            attention please!
            将列表introduction_split直接打印出来和单独打印introduction_split[1]、introduction_split[2]结果并不相同
            二者结果分别为
                        导演: 弗兰克·德拉邦特 Frank Darabont\xa0\xa0\xa0主演: 蒂姆·罗宾斯 Tim Robbins /...
                        导演: 弗兰克·德拉邦特 Frank Darabont   主演: 蒂姆·罗宾斯 Tim Robbins /...
            原因至今还未找到
            测试函数在最下面
        '''


# 保存数据函数
def save_data():
    f = open('douban_movie_top250.csv', 'a', newline='', encoding='utf-8-sig')
    global writer
    writer = csv.writer(f)
    writer.writerow(['电影名称', '评分', '评价人数', '引述', '导演/演员', '类别'])
    for i in range(10):
        url = 'https://movie.douban.com/top250?start=' + str(i * 25) + '&filter='
        response = requests.get(url, headers=headers)
        parse_html(response.text)
    f.close()


if __name__ == '__main__':
    save_data()



#测试函数
'''
import requests
from bs4 import BeautifulSoup

# 请求URL
url = 'https://movie.douban.com/top250'
# 请求头部
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'lxml')
movie_list = soup.find('ol', class_='grid_view').find_all('li')
introduction = movie_list[0].find('div', class_='bd').find('p', class_='').get_text()
print(introduction)
print(introduction.split('\n'))
introduction_split = introduction.split('\n')
for part in introduction_split:
    part = part.replace('\xa0', ' ')
print(introduction_split)
print('\n')
print('\n')
print(introduction_split[1])
print(introduction_split[2])
'''
