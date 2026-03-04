SAKey takes as input two parameters:
a) the file where you want the key discovery to take place
b) the number of exceptions n. 
Note that the n will lead to the discovery of n-non keys and then the extraction of (n-1)-almost keys



To run SAKey use the following command:
java -jar "jar_fullpath"/SAKey.jar "file_fullpath" n

For example, in the following command:
java -jar /Users/danai/SAKey.jar  /Users/danai/datasets/DB_Lake.nt 1

In this example, SAKey will discover first 1-non keys and then derive 0-almost keys for the file /Users/danai/datasets/DB_Lake.nt.


The file has to be in a n-triple format.
For example: 
<http://www.okkam.org/oaie/restaurant1-Restaurant56> <http://www.okkam.org/ontology_restaurant1.owl#phone_number> "718/522-5200" .
Or
http://www.okkam.org/oaie/restaurant1-Restaurant56	http://www.okkam.org/ontology_restaurant1.owl#phone_number	718/522-5200
