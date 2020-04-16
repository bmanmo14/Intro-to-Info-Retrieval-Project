import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.utility.Parameters;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

public class QueryResultsIO {

    private PrintStream writer;
    private static BufferedReader reader;

    public QueryResultsIO () throws UnsupportedEncodingException, FileNotFoundException {
        writer = null;
        reader = null;
    }

    public void newWriter(String filename) throws FileNotFoundException, UnsupportedEncodingException {
        if(writer == null){
            writer = new PrintStream(new BufferedOutputStream(new FileOutputStream(filename, false)), true, "UTF-8");
        }
    }

    public void writeLine(String queryNumber, HashMap<String, Double> results) {
        Set<String> document = results.keySet();
        int counter = 1;
        if (!results.isEmpty()) {
            for (String doc : document) {
                if(!Double.isNaN(results.get(doc))) {
                    writer.println(String.format("%s Q0 %s %d %6f galago", queryNumber, doc, counter, (double)results.get(doc)));
                    counter += 1;
                }
            }
        }
    }

    public HashMap<String, Double> writeLine(String queryNumber, List<ScoredDocument> results) {
        HashMap<String, Double> documentNames = new HashMap<>();
        if (!results.isEmpty()) {
            for (ScoredDocument sd : results) {
                writer.println(sd.toTRECformat(queryNumber));
                documentNames.put(sd.documentName, sd.score);
            }
        }
        return documentNames;
    }

    public void closeWriter(){
        if(writer != null){
            writer.close();
        }
        writer = null;
    }

    public static HashMap<String, HashMap<String, Double>> readFile(String inputFileName)  {
        HashMap<String, HashMap<String, Double>> result = new HashMap<>();
        try {
            File file = new File(inputFileName);
            reader = new BufferedReader(new FileReader(file));

            String st;
            while ((st = reader.readLine()) != null) {
                String[] words = st.split(" ");
                if (result.containsKey(words[0])) {
                    result.get(words[0]).put(words[2], Double.parseDouble(words[4]));
                } else {
                    HashMap<String, Double> newItem = new HashMap<String, Double>();
                    newItem.put(words[2], Double.parseDouble(words[4]));
                    result.put(words[0], newItem);
                }
            }
            reader.close();
        } catch (Exception e) {
            System.out.println(e);
        }
        return result;
    }

    public static Parameters createParameter(String queryID, String query, String type) throws IOException {
        switch (type){
            case "bm25":
                return Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"bm25(%s)\"}", queryID, query)));
            case "rm":
                return Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"#rm(%s)\"}", queryID, query)));
            case "jm":
                return Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"#jm(%s)\"}", queryID, query)));
            default:
                return Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"%s\"}", queryID, query)));
        }
    }

    public static List<Parameters> readParameters(String queryPath, String type){
        List<Parameters> queries = new ArrayList<Parameters>();
        try {
            File file = new File(queryPath);

            reader = new BufferedReader(new FileReader(file));

            String st;
            while ((st = reader.readLine()) != null) {
                String[] words = st.split("\t");
                queries.add(createParameter(words[0], words[1], type));
            }
            reader.close();
        } catch(Exception e){
            System.out.println(e);
        }
        return queries;
    }

    public static HashMap<String, String> createQueryPair(String queryPath) {
        HashMap<String, String> queriesPair = new HashMap<>();
        try {
            File file = new File(queryPath);
            reader = new BufferedReader(new FileReader(file));
            String st;
            while ((st = reader.readLine()) != null) {
                String[] words = st.split("\t");
                queriesPair.put(words[0], words[1]);
            }
            reader.close();
        } catch(Exception e){
            System.out.println(e);
        }
        return queriesPair;
    }
}
