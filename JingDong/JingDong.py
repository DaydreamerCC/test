from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

# 设置Chrome的WebDriver路径
chrome_options = Options()
chrome_options.add_argument("--headless")  # 无头模式，不打开浏览器界面
chrome_options.add_argument("--disable-gpu")

service = Service(executable_path="D:\chromedriver\chromedriver-win64\chromedriver.exe")  # 替换为你的chromedriver路径
driver = webdriver.Chrome(service=service, options=chrome_options)

# 京东商品搜索URL
search_url = "https://search.jd.com/Search?keyword=水杯&enc=utf-8"

# 访问搜索页面
driver.get(search_url)

# 让页面加载一段时间
time.sleep(3)

# 模拟滚动，加载更多商品
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)

# 获取商品信息
products = driver.find_elements(By.CLASS_NAME, 'gl-item')

# 存储商品信息的列表
product_list = []

# 遍历所有商品
for product in products:
    try:
        # 商品标题
        title = product.find_element(By.CSS_SELECTOR, 'div.p-name em').text

        # 商品价格
        price = product.find_element(By.CSS_SELECTOR, 'div.p-price strong i').text

        # 商品链接
        link = product.find_element(By.CSS_SELECTOR, 'div.p-name a').get_attribute('href')

        # 将信息添加到列表中
        product_list.append({
            'title': title,
            'price': price,
            'link': link
        })
    except Exception as e:
        print("Error parsing product:", e)

# 关闭浏览器
driver.quit()

# 将数据转换为DataFrame并保存到CSV文件
df = pd.DataFrame(product_list)
# df.to_csv("jd_products.csv", index=False)

# 打印爬取到的商品信息
print(df)
