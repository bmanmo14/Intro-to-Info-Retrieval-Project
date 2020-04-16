import org.lemurproject.galago.core.retrieval.ScoredDocument;

import java.io.*;
import java.util.*;


public class ResultWriter {
    private PrintStream out;
    
    public ResultWriter (String outputFileName, boolean append) throws UnsupportedEncodingException, FileNotFoundException{
        out = new PrintStream(new BufferedOutputStream(new FileOutputStream(outputFileName, append)), true, "UTF-8");
    }

    public HashMap<String, HashMap<String, Double>> readFile(String filePath) {
        HashMap<String, HashMap<String, Double>> result = new HashMap<>();
        try {
            File file = new File(filePath);

            BufferedReader br = new BufferedReader(new FileReader(file));

            String st;
            while ((st = br.readLine()) != null) {
                String[] words = st.split(" ");
                if (result.containsKey(words[0])) {
                    result.get(words[0]).put(words[2], Double.parseDouble(words[4]));
                } else {
                    HashMap<String, Double> newItem = new HashMap<String, Double>();
                    newItem.put(words[2], Double.parseDouble(words[4]));
                    result.put(words[0], newItem);
                }
            }
        } catch (Exception e) {
            System.out.println(e);
        }
        return result;
    }

    public void write (String queryNumber, HashMap<String, Double> results) {
        Set<String> document = results.keySet();
        int counter = 1;
        if (!results.isEmpty()) {
            for (String doc : document) {
                if(!Double.isNaN(results.get(doc))) {
                    out.println(String.format("%s Q0 %s %d %6f galago", queryNumber, doc, counter, (double)results.get(doc)));
                    counter += 1;
                }
            }
        }
    }
    
    public HashMap<String, Double> write (String queryNumber, List<ScoredDocument> results, boolean trecFormat) {
        HashMap<String, Double> documentNames = new HashMap<>();
        if (!results.isEmpty()) {
            for (ScoredDocument sd : results) {
                if (trecFormat) {
                    out.println(sd.toTRECformat(queryNumber));
                } else {
                    out.println(sd.toString(queryNumber));
                }
                documentNames.put(sd.documentName, sd.score);
            }
        }
        return documentNames;
    }
    
    public HashMap<String, Double> write (String queryNumber, List<ScoredDocument> results) {
        return write(queryNumber, results, true);
    }
    
    public void close (){
        out.close();
    }
}
