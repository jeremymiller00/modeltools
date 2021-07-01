import json
import requests
import time
import boto3
import s3fs
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from email import encoders
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import date
from pyspark.sql.types import *
from pyspark.sql.functions import *


class WoSRL_AB_tester:
    """
    A/B Tester Helper Class for WoSRL
    Takes raw data from Publons and S1 and prepares it for AB Testing
    """

    def __init__(self, raw_publons_data_a, raw_publons_data_b, raw_s1_data):
        publons_cols = ['datetime',
                        'recommendations',
                        'scholarOneRequestId',
                        'searchToken',
                        'searchUrl',
                        'version']
        s1_cols = ['assigned',
                   'completed ',
                   'config',
                   'datetime_decisioned',
                   'documentid',
                   'email_address',
                   'name',
                   'publons_search_result_url',
                   'selected',
                   'short_name']
        if self._check_columns(raw_publons_data_a, publons_cols):
            self.raw_publons_data_a = raw_publons_data_a
        else:
            raise ValueError("Data column mismatch; check schema for first dataset arg passed")
        if self._check_columns(raw_publons_data_b, publons_cols):
            self.raw_publons_data_b = raw_publons_data_b
        else:
            raise ValueError("Data column mismatch; check schema for second dataset arg passed")
        if self._check_columns(raw_s1_data, s1_cols):
            self.raw_s1_data = raw_s1_data
        else:
            raise ValueError("Data column mismatch; check schema for third dataset arg passed")
        self.a_selections_data = None
        self.b_selections_data = None
        self.a_completions_data = None
        self.b_completions_data = None
        self.stats = None

    def __repr__(self):
        if self.stats:
            return f"WoSRL AB Tester Stats:\n{self.stats.__repr__()}"
        else:
            return "WoSRL AB Tester: No stats yet available"

    def _check_columns(self, data, cols):
        return sorted(data.columns) == sorted(cols)

    def process_raw_data(self):
        # sets four processed data attributes
        a_publons_parsed = self._parse_publons(self.raw_publons_data_a)
        b_publons_parsed = self._parse_publons(self.raw_publons_data_b)
        s1_parsed = self._parse_s1(self.raw_s1_data)
        a_urls, b_urls = self._get_s1_urls(a_publons_parsed, b_publons_parsed)
        a_s1_parsed, b_s1_parsed = self._get_s1_splits(s1_parsed, a_urls, b_urls)
        self.a_selections_data = self._join_data(a_s1_parsed, a_publons_parsed, completions=False)
        self.b_selections_data = self._join_data(b_s1_parsed, b_publons_parsed, completions=False)
        self.a_completions_data = self._join_data(a_s1_parsed, a_publons_parsed, completions=True)
        self.b_completions_data = self._join_data(b_s1_parsed, b_publons_parsed, completions=True)

    def _parse_publons(self, data):
        # will be called twice, for a and b
        parsed = data \
            .select(trim(col("searchUrl")).alias("url"), explode(col("recommendations.email")).alias("_email")) \
            .withColumn("rec_email", trim(lower(col("_email")))) \
            .drop("_email") \
            .withColumn("recommended", lit(1))

        return parsed

    def _parse_s1(self, data):
        s1_parsed = data.select(trim(col("publons_search_result_url")).alias("publons_url"),
                                           trim(lower(col("email_address"))).alias("sel_email"),
                                           col("selected"),
                                           col("completed ").alias("completed"))
        return s1_parsed

    def _get_s1_urls(self, a_parsed, b_parsed):
        # get urls from Publons (searches) and filter to where s1 data matches these
        # there are lots of s1 decisioned manuscripts that don't make use of WoSRL
        a_urls = a_parsed.select("url").distinct()
        b_urls = b_parsed.select("url").distinct()
        return a_urls, b_urls

    def _get_s1_splits(self, s1_parsed, a_urls, b_urls):
        a_s1_parsed = s1_parsed.join(a_urls, s1_parsed["publons_url"] == a_urls["url"], "inner").drop("url")
        b_s1_parsed = s1_parsed.join(b_urls, s1_parsed["publons_url"] == b_urls["url"], "inner").drop("url")
        return a_s1_parsed, b_s1_parsed

    def _join_data(self, s1_data, publons_data, completions=False):
        if completions:
            s1_data = s1_data.filter(col("completed") == "Y")

        joined = s1_data.join(publons_data, \
                             (s1_data["publons_url"] == publons_data["url"]) & \
                             (s1_data["sel_email"] == publons_data["rec_email"]), \
                             "left_outer" \
                             ) \
            .na.fill(0, subset=["recommended"])
        return joined

    def calculate_stats(self):
        stats = {
            "a_sel": self.a_selections_data.count(),
            "a_sel_rec": self.a_selections_data.filter(col("recommended") == 1).count(),
            "b_sel": self.b_selections_data.count(),
            "b_sel_rec": self.b_selections_data.filter(col("recommended") == 1).count(),
            "a_comp": self.a_completions_data.count(),
            "a_comp_rec": self.a_completions_data.filter(col("recommended") == 1).count(),
            "b_comp": self.b_completions_data.count(),
            "b_comp_rec": self.b_completions_data.filter(col("recommended") == 1).count()
        }
        self.stats = stats

    def save_stats(self, path):
        with open(path, "w") as write_file:
            json.dump(self.stats, write_file)

class RCTester():
    def __init__(self, serviceAPI_A, serviceAPI_B, modelVersion_A, modelVersion_B):
        """
        A class to embody testing functionality for Reviewer Connect model deployed to dev endpoint.

        """
        if serviceAPI_A not in ["snapshot", "stable"] or serviceAPI_B not in ["snapshot", "stable"]:
            raise ValueError("""Invalid api reference; only valid options are "snapshot" or "stable" """)
        self.base_url_A = dbutils.secrets.get(scope="rltraining", key=serviceAPI_A + "url")
        self.base_url_B = dbutils.secrets.get(scope="rltraining", key=serviceAPI_B + "url")
        self.headers_A = {
          'X-ApiKey': dbutils.secrets.get(scope="rltraining", key=serviceAPI_A + "key"),
          'content-type': 'application/json'
        }
        self.headers_B = {
          'X-ApiKey': dbutils.secrets.get(scope="rltraining", key=serviceAPI_B + "key"),
          'content-type': 'application/json'
        }
        self.serviceAPI_A = serviceAPI_A
        self.serviceAPI_B = serviceAPI_B
        self.model_version_A = modelVersion_A
        self.model_version_B = modelVersion_B
        self.test_data = None
        self.api_data = None
        self.api_response = None
        self.response_DF = None
        self.metrics_DF = None
        self.metrics_avgs = None
        self.tp = None
        self.tp_fn = None
        self.recall = None
        self.s3_path = "s3://databricks-cc/reviewer-locator-data/model-api-test-results/"
        self.from_addr = "jeremy.miller@clarivate.com"
        self.to_addrs = [
            "jeremy.miller@clarivate.com"
        ]
        self.subject = "Reviewer Connect Recall at 30 Report"

    def load_and_preprocess_data(self, path: str, nOrdersNeeded: int, sub_sample: bool = False, verbose: int = 0):
        """
        Load sampled Publons test data and prepare for api
        There are ~1900 test samples
        A sample size of ~1000 should be adequate

        :param path: {str}: s3 path for the test data
        :param nOrdersNeeded: {int}: number of samples required for test. Spark sampling is approximate
        :param sub_sample: {bool}: how much to downsample the requests
        :param verbose: {bool}: how much to talk
        :return:
        """
        self._load_data(path)
        if verbose > 0:
            print("Raw data loaded")
        if sub_sample:
            fraction = self._get_sample_size(nOrdersNeeded)
        else:
            fraction = 1
        if verbose > 0:
            print("Raw data downsampled")
        self._get_api_data(fraction)
        if verbose > 0:
            print("Load and preprocess complete")
        if verbose > 1:
            print(f"{len(self.api_data)} Records Preprocessed")

    def _load_data(self, path:str):
        """
        Simple wrapper for loading Spark DataFrame
        Lowercase the email addresses from S1
        
        :param path:
        :return:
        """
        df = spark.read.json(path)
        test_data = df.drop("datetime", "scholarOneRequestId", "searchToken" "searchUrl",) \
            .withColumn("email", explode("reviewedOrSelected")) \
            .withColumn("email_lower", lower(col("email"))) \
            .withColumn("email_split", split(col("email_lower"), ";")) \
            .withColumn("email_formatted", explode("email_split")) \
            .drop("email", "email_lower", "email_split", "reviewedOrSelected") \
            .groupBy("dataScienceRequestData", "publons_search_result_url").agg(collect_list(col("email_formatted")).alias("reviewedOrSelected"))
        self.test_data = test_data

    def _get_sample_size(self, nOrdersNeeded: int) -> float:
        """
        Find sample fraction needed to achieve desired sample size
        nb: Spark sampling is approximate

        :param nOrdersNeeded: {int}: number of samples required for test. Spark sampling is approximate
        :return: {float}: fraction needed
        """
        n_orders = self.test_data.select(col("publons_search_result_url")).distinct().count()
        # account for approximate-sampling variance
        fraction = (nOrdersNeeded / n_orders) * 1.2
        return fraction

    def _get_api_data(self, fraction: int):
        """
        Convert df to json format for posting to api

        :param fraction: {float}: fraction needed
        :return:
        """
        if fraction < 1:
            api_df = self.test_data.sample(fraction, seed=123) \
                .select("dataScienceRequestData", "publons_search_result_url") \
                .distinct()
        elif fraction == 1:
            api_df = self.test_data \
                .select("dataScienceRequestData", "publons_search_result_url") \
                .distinct()
        else:
            raise ValueError("Sample Proportion Error")
        self.api_data = api_df.toJSON().map(lambda j: json.loads(j)).collect()
    
    def get_response(self, num_recommendations: int,  verbose: int = 0):
        """
        Send search request to the API and return the results.
        Parameters:
        ----------
        input: 
        data {list[dict]}: the data for posting to the api
        num_recommendations {int}: number of recommended journals for each manuscript
        api_version {str}: environment of the api to post to
        model_version {str}: version of the model to use; unique to Reviewer Connect
        verbose {bool}: print progress every 10 records, the default is 100 
        
        output: {dict}: responses from the api in a next dictionary structure; each element in the list represents the results of one search
        """
        output = []
        n = 0
        invalid_email_count = 0
        recommendation_count = 0
        score_count = 0
        invalid_json = 0
        start_time = int(time.time())
        url_A = self.base_url_A + '?ver=' + self.model_version_A
        url_B = self.base_url_B + '?ver=' + self.model_version_B

        for item in self.api_data:
            n += 1
            payload = item['dataScienceRequestData']
            payload['reviewerLocatorRequiredMatch'] = num_recommendations
            post_data = json.dumps(payload)
            response_A = self._post_data(url=url_A, headers=self.headers_A, data=post_data)
            response_B = self._post_data(url=url_B, headers=self.headers_B, data=post_data)
            try:
                json_response_A = response_A.json()
                json_response_B = response_B.json()
                package = {'search_url': item['publons_search_result_url']}
                recommendations_A = []
                scores_A = []
                recommendations_B = []
                scores_B = []
                
                for rec in json_response_A['recommendations']:
                    if isinstance(rec['reviewer']['email'], str):
                        recommendations_A.append(rec['reviewer']['email'].lower())
                        recommendation_count += 1
                    else:
                        print("Invalid email for recommendation. Continuing...")
                        invalid_email_count += 1
                    if isinstance(rec['debug']['score'], float):
                        scores_A.append(rec['debug']['score'])
                        score_count += 1                      
                package['recommendations_A'] = recommendations_A
                package['scores_A'] = scores_A
                
                for rec in json_response_B['recommendations']:
                    if isinstance(rec['reviewer']['email'], str):
                        recommendations_B.append(rec['reviewer']['email'].lower())
                        recommendation_count += 1
                    else:
                        print("Invalid email for recommendation. Continuing...")
                        invalid_email_count += 1
                    if isinstance(rec['debug']['score'], float):
                        scores_B.append(rec['debug']['score'])
                        score_count += 1                      
                package['recommendations_B'] = recommendations_B
                package['scores_B'] = scores_B
                
                output.append(package)
                
                if verbose == 2:
                    elapsed_time = int(time.time()) - start_time
                    print("Processed item {} of {} after {} seconds...".format(n, len(self.api_data), elapsed_time))
                elif verbose == 1:
                    if n % 10 == 0:
                        elapsed_time = int(time.time()) - start_time
                        print("Processed item {} of {} after {} seconds...".format(n, len(self.api_data), elapsed_time))
                else:
                    if n % 100 == 0:
                        elapsed_time = int(time.time()) - start_time
                        print("Processed item {} of {} after {} seconds...".format(n, len(self.api_data), elapsed_time))
            except:
                print("Invalid json in response. Continuing...")
                invalid_json += 1
                
        print("Done!")
        print(f"Recommendations with email addresses: {recommendation_count}")
        print(f"Recommendations without email addresses: {invalid_email_count}")
        print(f"Recommendations with scores: {score_count}")
        print(f"Invalid json responses: {invalid_json}")
        self.api_response = output

    def _post_data(self, url, headers, data, n_tries=3):
        for i in range(n_tries):
            response = requests.post(url=url, headers=headers, data=data)
            if response.status_code != 200:
                if i == n_tries-1:
                    print(
                        f"Request to url A failed with status {response.status_code}, {response.text} after 3 tries")
                continue
            else:
                break
        return response

    def post_process_data(self):
        """
        Use api response to calculate recall at 30
        """
        self._response_to_df()
        self._calculate_comparison_metrics()

    def _response_to_df(self):
        """
        Transform raw api response json to PySpark DataFrame
        """
        response_json = sc.parallelize(self.api_response)
        response_DF = sqlContext.read.json(response_json)
        self.response_DF = response_DF.toPandas()

    def _calculate_comparison_metrics(self):
        df = self.response_DF.copy()
        df['min_A'] = df['scores_A'].apply(lambda x: np.min(x))
        df['min_B'] = df['scores_B'].apply(lambda x: np.min(x))
        df['med_A'] = df['scores_A'].apply(lambda x: np.median(x))
        df['med_B'] = df['scores_B'].apply(lambda x: np.median(x))
        df['max_A'] = df['scores_A'].apply(lambda x: np.max(x))
        df['max_B'] = df['scores_B'].apply(lambda x: np.max(x))
        df['min_A > min_B'] = df['min_A'] > df['min_B']
        df['med_A > med_B'] = df['med_A'] > df['med_B']
        df['max_A > max_B'] = df['max_A'] > df['max_B']
        df = df[['min_A', 'min_B', 'med_A', 'med_B', 'max_A', 'max_B', 'min_A > min_B', 'med_A > med_B',
                   'max_A > max_B']]
        avgs = df.mean()
        self.metrics_avgs = pd.DataFrame(avgs, columns=['Average Value'])
        self.metrics_DF = df

    def _calculate_recall(self):
        """
        Calculate recall at 30 from api response
        """
        data_joined = self.response_DF.join(self.test_data, 
                                          self.response_DF['search_url'] == self.test_data['publons_search_result_url'], 
                                          'inner') \
            .drop(self.response_DF['search_url'])
              
        with_recall = data_joined.withColumn("found", array_intersect("recommendations", "reviewedOrSelected")) \
            .withColumn("recall", size("found") / size("reviewedOrSelected"))
      
        with_sums = with_recall.select(size("found").alias("sum_found"), size("reviewedOrSelected").alias("sum_reviewed_selected")).groupby().sum()

        self.tp = with_sums.collect()[0][0]
        self.tp_fn = with_sums.collect()[0][1]
        self.recall = self.tp / self.tp_fn
        
    def save_metrics_data(self):
        if not isinstance(self.metrics_DF, pd.DataFrame):
            print("No metric data available")
            return
        self.metrics_DF.to_csv(self.s3_path + str(int(time.time())) + "-metrics.csv")

    def send_email(self, attachments=[], cc=[], bcc=[], verbose=0):
        attachment_ready_html = []
        img_id = 0
        mime_images = []
        # iterate over raw HTML
        html = self._get_html()
        for l in html:
            # For each image in the body_html, convert each to a base64 encoded inline image
            if l.startswith("<img"):
                image_data = l[len("<img src='data:image/png;base64,"):-2]
                mime_img = MIMEImage(base64.standard_b64decode(image_data))
                mime_img.add_header('Content-ID', '<img-%d>' % img_id)
                attachment_ready_html.append("<center><img src='cid:img-%d'></center>" % img_id)
                img_id += 1
                mime_images.append(mime_img)
            else:
                attachment_ready_html.append(l)
        print("Added {} images".format(img_id))

        msg = MIMEMultipart()
        msg['Subject'] = self.subject
        msg['From'] = self.from_addr
        msg['To'] = ", ".join(self.to_addrs)
        body = MIMEText('\n'.join(attachment_ready_html), 'html')
    
        for i in mime_images:
            msg.attach(i)
      
        msg.attach(body)
    
        for raw_attachment in attachments:
            attachment = MIMEApplication(open(raw_attachment, 'rb').read())
            attachment.add_header('Content-Disposition', 'attachment', filename=raw_attachment)
            msg.attach(attachment)
    
        ses = boto3.client('ses', region_name='us-west-2')
        ses.send_raw_email(
            Source=msg['FROM'],
            Destinations=self.to_addrs,
            RawMessage={'Data': msg.as_string()})    
        print("Sending Email.")
        if verbose > 0:
            print(f"Email sent to: {self.to_addrs}")
        
    def _get_html(self):
        html = [
            "<center><h1>Reviewer Connect Recall at 30 Report</h1></center>",
            f"""
            <p><b>Metrics: {self.metrics_avgs.to_html()}.</b></p>
            <p><b>Test Run on {date.today()}.</b></p>
            <p><b>Model Version A = {self.serviceAPI_A} - {self.model_version_A}.</b></p>
            <p><b>Model Version B = {self.serviceAPI_B} - {self.model_version_B}.</b></p>
          """
        ]
        return html


def calculate_binom_metric(trials: int, successes: int, metric_name: str, cred_int: int = 95) -> plt.figure:
    """    
    Wrapper function for calculating and plotting relevant metrics.
    Uses Bayesian updating to produce a Beta distribution of the relevant metric.
    Appropriate for any metric that can be modeled by a binomial distribution:
        k successes in N trials

    Params:
    trials: number of relevant labels
    successes: number of 'successful' labels, definition varies by metric, see scoping doc for definitions
    metric_name: Accuracy, Recall, Precision, Readability
    cred_int: desired size of credible interval
    
    Returns:
    Labeled fig with relevant metrics
    
    """
    failures = trials - successes
    posterior = stats.beta(1 + successes, 1 + failures)
    sample = posterior.rvs(size=10000)    
    bootstrap_ci = np.percentile(sample, [100-cred_int, cred_int])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.linspace(0, 1, 10000)
    y = posterior.pdf(x) / trials
    idx = y.argmax()
    expected_value = x[idx]
    
    ax.plot(x, y)
    ax.vlines(x=expected_value, ymin=0, ymax=posterior.pdf(expected_value) / trials, color='orange')
    ax.set_xlabel(metric_name)
    ax.set_ylabel("PDF")
    ax.grid()
    ax.set_xlim([0.5, 1.0])
    title_1 = f"Probability Distribution of {metric_name} for RF Labels"
    title_2 = f"Most Likely {metric_name} Value: {perc(expected_value)}%"
    title_3 = f"N = {trials}"
    title_4 = f"{cred_int}% {metric_name} Credible Interval: {perc(bootstrap_ci[0])}% -> {perc(bootstrap_ci[1])}%"
    ax.set_title(f"{title_1}\n{title_2}\n{title_3}\n{title_4}", fontsize=20)
    
    return fig

def perc(x: float) -> int:
    """
    Format a 0-1 decimal value to display as a percentage
    """
    return int(np.round(x * 100))