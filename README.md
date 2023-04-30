Download Link: https://assignmentchef.com/product/solved-dsl-lab3-frequent-itemsets-and-association-rules
<br>
In this laboratory, you will learn more about frequent itemsets and association rules. You will first use an existing library that implements Apriori and FP-Growth. Then, you will implement your own version of

Apriori.

<h1>1          Preliminary steps</h1>

<h2>1.1         mlxtend and pandas</h2>

Make sure you have these two libraries installed. You can check whether a given library is installed by importing it in Python.

import mlxtend import pandas

Or, if you use it, by checking pip:

$ pip freeze | egrep “(mlxtend|pandas)” mlxtend==0.17.0 pandas==0.25.1

If not available, you will need to install them with pip install [package] (or any other package manager you may be using).

Pandas is a library for easy-to-use data structures and data analysis tools in Python. You will be learning about this library later in this course but, for the time being, you will use some basic functionalities, as it is required to use Mlxtend.

Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks. For most of the course we will be using scikit-learn. However, Mlxtend implements various frequent itemsets algorithms that are missing from scikit-learn. As such, for this laboratory, we will be using it. You can find out more about Mlxtend on the <a href="https://rasbt.github.io/mlxtend/">official website</a><a href="https://rasbt.github.io/mlxtend/">.</a>

<h2>1.2         Datasets</h2>

For this lab, two different datasets will be used. Here, you will learn more about them and how to retrieve them.

<h3>1.2.1         Online Retail Data Set</h3>

The Online Retail Data Set [1] is a dataset made available on the <a href="https://archive.ics.uci.edu/ml/datasets/online+retail">UCI Machine Learning repository</a><a href="https://archive.ics.uci.edu/ml/datasets/online+retail">.</a> It which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based online retail.

The original version of the dataset is available on UCI ML as a .xlsx. For your convenience, we are also making it available as a CSV file at the following URL. https://github.com/dbdmg/data-science-lab/raw/master/datasets/online_retail.csv

Each of the 541,909 rows contains an item that has been purchased by someone. Items can be grouped into invoices (you can think of these as receipts), where each invoice has been issued for a specific buyer, and can contain multiple items.

The columns contained in the CSV file are the following:

<ul>

 <li>InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter §c’, it indicates a cancellation.</li>

 <li>StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.</li>

 <li>Description: Product (item) name. Nominal.</li>

 <li>Quantity: The quantities of each product (item) per transaction. Numeric.</li>

 <li>InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.</li>

 <li>UnitPrice: Unit price. Numeric, Product price per unit in sterling.</li>

 <li>CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.</li>

 <li>Country: Country name. Nominal, the name of the country where each customer resides. As an example, the following lines all refer to the same invoice (InvoiceNo = 574021).</li>

</ul>

434006,574021,23301,GARDENERS KNEELING PAD KEEP CALM ,48,2011-11-02 12:18:00,1.45,14434.0,United Kingdom

434007,574021,23355,HOT WATER BOTTLE KEEP CALM,24,2011-11-02 12:18:00,4.15,14434.0,United Kingdom

434008,574021,23284,DOORMAT KEEP CALM AND COME IN,20,2011-11-02 12:18:00,7.08,14434.0,United Kingdom

<h3>1.2.2         COCO Dataset</h3>

<a href="http://cocodataset.org/">COCO Dataset</a> is a large-scale object detection, segmentation, and captioning dataset. It offers a large number of images (from various contexts) with annotations (i.e. structured information on the contents of the image). These annotations, in particular, regard the contents of the image and, in particular, the objects contained within.

You can download a filtered and preprocessed version of COCO (which we will refer to as “modified”) from the following URL:

https://raw.githubusercontent.com/dbdmg/data-science-lab/master/datasets/modified_coco.json This dataset is a JSON file. You can open it using the already introduced json module. The file contains a list of images and, for each image, the annotation key contains all the annotations available. The following are the annotations for one such image.

{

“file_name”: “000000465265.png”,

“image_id”: 465265,

“annotations”: [

“person”,

“person”,

“person”,

“fire hydrant”,

“handbag”,

“chair”,

“cell phone”

]

}

This means that the image contains 3 people, a fire hydrant, a handbag, a chair and a cell phone. Indeed the actual image is represented in Figure 1 (with and without annotations visualized (also known as “segments”)). You can can access each image (identified by the image_id) on the website, at: <a href="http://cocodataset.org/#explore">http://cocodataset.org/explore</a><a href="http://cocodataset.org/#explore">.</a>

(a) Not segmented                                                                          (b) Segmented

Figure 1: Image #465265, with and without segmentation

<h1>2          Exercises</h1>

Note that exercises marked with a (*) are optional, you should focus on completing the other ones first.

<h2>2.1         Association rules from frequent itemsets</h2>

This exercise will work on the Online Retail Data Set. In particular, you will do some data preprocessing on the dataset to extract all itemsets available (where each itemset is the collection of items contained in a single invoice). Then, using FP-Growth and Apriori implementations, you will extract a list of frequent itemsets. From those, you will finally extract several different association rules.

<ol>

 <li>First, you need to load the dataset into memory, using the csv module. Make sure you identify all valid rows. Also consider that rows having an InvoiceNo that starts with C should be discarded, as they indicate that the invoice is about a cancelled purchase.</li>

 <li>Now that you have a dataset of items, you should aggregate it at an “invoice” level. For each invoice (identified by InvoiceNo) there can be multiple items (from multiple rows) in the dataset. For each invoice, you should build a list of all items belonging to it. For the example invoice presented in 2.1, you want to build the following list:</li>

</ol>

[ “GARDENERS KNEELING PAD KEEP CALM”,

“HOT WATER BOTTLE KEEP CALM”,

“DOORMAT KEEP CALM AND COME IN” ]

<ol start="3">

 <li>You should now have a list (one for each invoice) of lists (each list containing the items bought for that invoice). Now, we need to convert this into a matrix form. Of the many possible formats, we will use the one expected by the Mlxtend library, which is as follows. Given an ordered list of <em>M </em>possible items (in this case, all possible products that can be bought), and given <em>N </em>itemsets (in this case, invoices), we should build a matrix of <em>N </em>rows and <em>M </em> The element at the <em>i<sup>th </sup></em>row and <em>j<sup>th </sup></em>column should be 1 if the <em>i<sup>th </sup></em>itemset (invoice) contains the <em>j<sup>th </sup></em>item (product), 0 otherwise. For the following example:</li>

</ol>

a,b,c b,c a,c,d a,b

The list of all possible items is [a, b, c, d]. As such, the matrix that we will build is the following:

1 1 1 0

<ul>

 <li>1 1 0</li>

 <li>0 1 1</li>

</ul>

1 1 0 0

Once we have defined this matrix (as a list of lists), we can use Pandas to convert it to a <em>DataFrame </em>(which is, essentially, a table) with the following code:

import pandas as pd

all_items = [‘a’, ‘b’, ‘c’, ‘d’] # this is your list of items pa_matrix = [

[1,1,1,0],

[0,1,1,0],

[1,0,1,1],

[1,1,0,0]

] # this is the matrix you built from the itemsets df = pd.DataFrame(data=pa_matrix, columns=all_items)

<ol start="4">

 <li>With the df that you defined in the previous exercise, you can now use the fp_growth function. This function, which is described in the detail in the <a href="https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#fpgrowth">official documentation</a><a href="https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#fpgrowth">.</a> The first argument required is the previously built DataFrame, df. The second is the minimum support (<em>minsup</em>), i.e. the minimum fraction of the entire dataset in which the itemset should show up for it to be considered “frequent”. Try using different values of <em>minsup</em>, such as 0.5, 0.1, 0.05, 0.02, 0.01. How many results do you obtain as <em>minsup </em>varies? You can check the number of frequent itemsets identified and print them all with the following code snipped:</li>

</ol>

fi = fpgrowth(df, 0.05) print(len(fi)) print(fi.to_string())

<ol start="5">

 <li>Consider the itemsets extracted for <em>minsup </em>= 0.02. How many items are contained? Which ones would you be considered the most useful?</li>

 <li>Use the value returned by fpgrowth to extract the relevant association rules.</li>

 <li>Extract the association rules from the frequent itemsets extracted with <em>minsup </em>= 0.01. You can find the documentation for association_rules() on the <a href="https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#association_rules">official documentation</a><a href="https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#association_rules">.</a> You can use the confidence as the metric to identify the rules, and a minimum threshold of 0.85 (feel free to vary these values and observe how the results vary).</li>

 <li>(*) Rerun the experiments from point 4 with apriori() (documentation on the <a href="https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#apriori">official website</a><a href="https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/#apriori">)</a>. Do the results match with the ones found by FP-Growth? Is Apriori faster or slower than FP-Growth? You can measure how long a function call takes with the following code snippet:</li>

</ol>

import timeit

# number=1 means that it executes the function only once timeit.timeit(lambda: apriori(df, 0.01), number=1)

<h2>2.2         Apriori implementation</h2>

In this exercise, you will implement your own version of Apriori and use it on the COCO dataset to extract frequent itemsets (i.e. groups of annotations that often co-occur within the same image, such as ’car’ and

’traffic light’).

Note that, while this entire exercise is optional, we recommend you try to solve it anyway. You may argue that there are libraries already implementing these and other algorithms. Despite that, we believe it is important, for a data scientist, to know the underlying theory as well as some implementation details. You can learn the former on textbooks, but you will only be faced with the latter when actually working on the implementation.

<ol>

 <li>You can find a thorough explanation of the Apriori algorithm on the course slides on <a href="http://dbdmg.polito.it/wordpress/wp-content/uploads/2019/10/DSL-3-MassRules.pdf">association </a><a href="http://dbdmg.polito.it/wordpress/wp-content/uploads/2019/10/DSL-3-MassRules.pdf">rules</a> (slides 16-20). With these, implement your own version of Apriori. You can use the below toy dataset from the example in slides 21-35 for initial troubleshooting and testing.</li>

</ol>

a,b b,c,d a,c,d,e a,d,e a,b,c a,b,c,d b,c a,b,c a,b,d b,c,e

When run with <em>minsup </em><em>&gt; </em>1 (or 0.1 in relative terms), the expected itemset (with their <em>minsup</em>s) are:

a -&gt; 7 b -&gt; 8 c -&gt; 7 d -&gt; 5 e -&gt; 3 a,b -&gt; 5

a,c -&gt; 4

a,d -&gt; 4

a,e -&gt; 2

b,c -&gt; 6

b,d -&gt; 3

c,d -&gt; 3

c,e -&gt; 2 d,e -&gt; 2

a,b,c -&gt; 3

a,b,d -&gt; 2

a,c,d -&gt; 2

a,d,e -&gt; 2

b,c,d -&gt; 2

<ol start="2">

 <li>Once you have implemented a working version of Apriori, you can load the modified COCO dataset from Subsection 2.2 into memory. From this, you should transform the dataset into a version compatible with the expected input of your Apriori implementation.</li>

 <li>You can now run your implementation on the modified COCO dataset. Try using a <em>minsup </em>of 0.02, as well as other values. Are the obtained results meaningful? You can use the COCO dataset “explore” tool to examine any of the image used.</li>

 <li>Now convert the modified COCO dataset into the format required by Mlxtend’s apriori() and fpgrowth() functions (i.e. the one described in Exercise 3). Then, run these two algorithms on this dataset. Do the results from these functions match your results? (Hint: they should!)</li>

 <li>How do apriori() and fpgrowth() compare in terms of execution time with your implementation? You can use the previously introduced timeit library to run comparisons.</li>

 <li>(*) Finally, you can try with an exhaustive approach to finding frequent itemsets. This approach consists in generating all possible itemsets of length ≤ <em>k </em>and, for each one, counting the number of occurrences within the dataset. Try running it with low values of <em>k </em>(start from 1). How long does that take? How does the number of possible itemsets grow with <em>k</em>? Would that be feasible?</li>

</ol>