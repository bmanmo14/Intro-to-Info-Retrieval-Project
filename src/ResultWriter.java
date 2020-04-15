import org.lemurproject.galago.core.retrieval.ScoredDocument;

import java.io.*;
import java.util.*;


public class ResultWriter {
    private PrintStream out;
    
    public ResultWriter (String outputFileName, boolean append) throws UnsupportedEncodingException, FileNotFoundException{
        out = new PrintStream(new BufferedOutputStream(new FileOutputStream(outputFileName, append)), true, "UTF-8");
    }
    
    public ResultWriter (){
        out = System.out;
    }

    public void write (String queryNumber, HashMap<String, Double> results) {
        Set<String> document = results.keySet();
        int counter = 1;
        if (!results.isEmpty()) {
            for (String doc : document) {
                if(!Double.isNaN(results.get(doc))) {
                    out.println(String.format("%s Q0 %s %2d %6f galago", queryNumber, doc, counter, (double)results.get(doc)));
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
