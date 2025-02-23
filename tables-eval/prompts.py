HTML_PROMPT = """Extract the data from the table below and provide the output in HTML format. Output only the data as HTML and nothing else. Here is one example:
```html
<table> <thead> <tr> <th>الفئة</th> <th>النسبة المئوية</th> <th>التفاصيل</th> </tr> </thead> <tbody> <tr> <td>الأسهم المحلية</td> <td>٣٥٪</td> <td>شركة سابك, شركة الاتصالات السعودية, شركة أرامكو</td> </tr> <tr> <td>الأوراق المالية الحكومية</td> <td>٢٠٪</td> <td>حكومة السعودية, حكومة الإمارات</td> </tr> <tr> <td>السندات الدولية</td> <td>١٥٪</td> <td>بنك سويسري, بنك جي بي مورغان</td> </tr> <tr> <td>العقارات التجارية</td> <td>١٥٪</td> <td>دبي, الرياض, المنامة</td> </tr> <tr> <td>الاستثمارات البديلة</td> <td>١٠٪</td> <td>صناديق الاستثمار الخاصة, صناديق التحوط</td> </tr> <tr> <td>النقد وما يعادله</td> <td>٥٪</td> <td>بنك الإمارات دبي الوطني, بنك أبوظبي الأول</td> </tr> </tbody> </table>
```
Now generate the data for the provided table."""

DF_PROMPT = """Extract the data from the table below and provide the output in CSV format. Output only the data as CSV and nothing else. Here is one example:
```csv
اسم الشركة,الصفقة,مبلغ الصفقة (مليون دولار),تاريخ الاتفاقية,نوع التقنية 
أوراكل,الاستحواذ على شركة سيرنر,28,2023-06-15,الحوسبة السحابية والنمذجة الحيوية 
أمازون ويب سيرفيسز,شراكة مع شركة موديلينغ بيو,15,2023-04-20,النمذجة الحيوية 
مايكروسوفت,شراكة مع شركة بيومادكس,12,2023-03-10,الحوسبة السحابية 
جوجل كلاود,شراء شركة بيوكيم سوليوشنز,35,2023-09-01,النمذجة الحيوية 
آي بي إم,توسع في شراكتها مع شركة جينوميك سوفتوير,18,2023-05-05,حوسبة بيولوجية

```
Now generate the data for the provided table."""