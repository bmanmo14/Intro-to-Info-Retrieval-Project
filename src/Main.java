/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
import edu.stanford.nlp.math.ArrayMath;
import org.apache.lucene.util.MathUtil;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.lemurproject.galago.contrib.retrieval.traversal.BM25FTraversal;
import org.lemurproject.galago.core.eval.Eval;
import org.lemurproject.galago.core.eval.QuerySetJudgments;
import org.lemurproject.galago.core.index.disk.DiskIndex;
import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.utility.MathUtils;
import org.lemurproject.galago.utility.Parameters;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.javatuples.Pair;

import java.io.*;
import java.lang.reflect.Parameter;
import java.util.*;

/**
 * @author raver119@gmail.com
 */
public class Main {
    public static final String indexPath = "index";
    public static final String queryPath = "PsgRobust/robust04.descs.tsv";
    public static final String queryJudgement = "PsgRobust/PsgRobust.qrels";
    public static final String wordvecPath = "/Users/brandonmouser/Downloads/GoogleNews-vectors-negative300.bin";
    public static final String queryWordVectorPath = "QUERY_WORDS.txt";
    public static final String dHatVectorPath = "D_HAT.txt";
    public static final String resultsPath = "batchResults";
    public static final String combinedResultsPath = "combinedResults";
    public static final String WEResultsPath = "WEResults";
    public static final String altResultsPath = "altResults";
    public static final int requested = 1000;

    public static List<Parameters> readParameters(boolean rm){
        List<Parameters> queries = new ArrayList<Parameters>();
        try {
            File file = new File(queryPath);

            BufferedReader br = new BufferedReader(new FileReader(file));

            String st;
            while ((st = br.readLine()) != null) {
                String[] words = st.split("\t");
                if(rm)
                    queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"#rm(%s)\"}", words[0], words[1]))));
                else
                    queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"%s\"}", words[0], words[1]))));
            }
        } catch(Exception e){
            System.out.println(e);
        }
        return queries;
    }

    public static Retrieval createRet(String stemmer, String stemmerClass, String textPart,
                                      String scorer, String model, int docs, int term, double weight, boolean mode){
        try {
            Parameters queryParams = Parameters.create();
            queryParams.set("index", indexPath);
            queryParams.set("verbose", true);
            queryParams.set("defaultTextPart", textPart);
            queryParams.set("stemmerClass", stemmerClass);
            queryParams.set("rmStemmer", "org.lemurproject.galago.core.parse.stem." + stemmer);
            queryParams.set("smoothMethod", scorer);
            queryParams.set("scorer", scorer);
            if(mode){
                queryParams.set("relevanceModel", model);
                queryParams.set("fbDocs", docs);
                queryParams.set("fbTerm", term);
                queryParams.set("fbOrigWeight", weight);
            }
            return RetrievalFactory.create(queryParams);
        } catch(Exception e){
            System.out.println(e);
        }
        return null;
    }
    public static void write_q() throws Exception {
        Word2Vec wordVectors = WordVectorSerializer.readBinaryModel(new File(wordvecPath), false, false);
        PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(queryWordVectorPath, false)), true, "UTF-8");

        List<Parameters> queries = readParameters(false);
        for(Parameters p : queries) {
            String[] query_string = p.getAsString("text").split(" ");
            for (String term : query_string) {
                if (wordVectors.hasWord(term)) {
                    out.print(term + " ");
                    double[] vector = wordVectors.getWordVector(term);
                    for (double item : vector) {
                        out.print(item);
                        out.print(" ");
                    }
                    out.println();
                }
            }
        }
        System.out.println("Processed queries");
        out.close();

    }

    public static void write_d_hat() throws Exception {
        Word2Vec wordVectors = WordVectorSerializer.readBinaryModel(new File(wordvecPath), false, false);
        Retrieval ret = createRet("KrovetzStemmer", KrovetzStemmer.class.getName(), "postings.krovetz",
                "dirichlet", "org.lemurproject.galago.core.retrieval.prf.RelevanceModel3", 20, 100, 0.25, true);
        FieldStatistics fs = ret.getCollectionStatistics ("#lengths:part=lengths()");
        DiskIndex index = new DiskIndex(indexPath);
        Document.DocumentComponents dc = new Document.DocumentComponents(false, false, true);
        PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(dHatVectorPath, false)), true, "UTF-8");

        for(int i = 0; i < fs.documentCount; i++) {
            String name = index.getName((long) i);
            List<String> terms = index.getDocument(name, dc).terms;
            double[] wordVector = new double[wordVectors.getWordVector("the").length];

            int count = 0;
            for(String term : terms){
                if(wordVectors.hasWord(term)) {
                    double[] vector = wordVectors.getWordVector(term);
                    double norm = ArrayMath.L2Norm(vector);
                    ArrayMath.divideInPlace(vector, norm);
                    wordVector = ArrayMath.pairwiseAdd(wordVector, vector);
                    count += 1;
                }
            }
            ArrayMath.divideInPlace(wordVector, (double) count);
            out.print(name + " ");
            for(double item : wordVector){
                out.print(item);
                out.print(" ");
            }
            out.println();
            System.out.print("\rProcessed document: " + String.valueOf(i) + "/" + String.valueOf(fs.documentCount));
        }
        out.close();
    }

    public static Parameters setParams(List<Pair> paramTuplesStr,
                                       List<Pair> paramTuplesBool,
                                       List<Pair> paramTuplesInt) {

        Parameters params = Parameters.create();
        for (Pair p : paramTuplesStr) {
            params.set(((String) p.getValue0()), ((String) p.getValue1()));
        }
        for (Pair p : paramTuplesBool) {
            params.set(((String) p.getValue0()), ((Boolean) p.getValue1()));
        }
        for (Pair p : paramTuplesInt) {
            params.set(((String) p.getValue0()), ((Integer) p.getValue1()));
        }
        return params;
    }

    public static List<String> getAllDocumentNames() throws Exception {
        List<String> documentNames = new ArrayList<String>();
        Retrieval ret = createRet("KrovetzStemmer", KrovetzStemmer.class.getName(), "postings.krovetz",
                "dirichlet", "org.lemurproject.galago.core.retrieval.prf.RelevanceModel3", 20, 100, 0.25, true);
        FieldStatistics fs = ret.getCollectionStatistics ("#lengths:part=lengths()");
        DiskIndex index = new DiskIndex(indexPath);
        for(int i = 0; i < fs.documentCount; i++) {
            documentNames.add(index.getName((long) i));
        }
        return documentNames;
    }

    private static void normalize(HashMap<String, HashMap<String, Double>> ...maps){
        for (HashMap<String, HashMap<String, Double>> queryMap : maps) {
            for (HashMap<String, Double> docMap : queryMap.values()) {
                double sum = 0;
                for (Double score : docMap.values())
                    sum += score;
                for (Map.Entry<String, Double> pair : docMap.entrySet())
                    docMap.put(pair.getKey(), pair.getValue() / sum);
            }
        }
    }

     private static HashMap<String, HashMap<String, Double>> combine(HashMap<String, HashMap<String, Double>> map1,
                                                                     HashMap<String, HashMap<String, Double>> map2,
                                                                     double alpha) {
        HashMap<String, HashMap<String, Double>> combinedMap = new HashMap<String, HashMap<String, Double>>();
        for (Map.Entry<String, HashMap<String, Double>> queryPair : map1.entrySet()) {
            String queryKey = queryPair.getKey();
            combinedMap.put(queryKey, new  HashMap<String, Double>());
            for (String docKey : queryPair.getValue().keySet()) {
                 double combinedVal = (1 / alpha) * map1.get(queryKey).get(docKey) + alpha * map2.get(queryKey).get(docKey);
                 combinedMap.get(queryKey).put(docKey, combinedVal);
            }
        }
        return combinedMap;
    }

    private static void writeResults(HashMap<String, HashMap<String, Double>> scoreMap, String path) throws Exception {
        QueryResultsIO resultWriter = new QueryResultsIO();
        resultWriter.newWriter(path);
        for (Map.Entry<String, HashMap<String, Double>> queryPair : scoreMap.entrySet()) {
            String queryKey = queryPair.getKey();
            HashMap<String, Double> sortedDocs = WordEmbeddedScorer.sortByValue(queryPair.getValue());
            resultWriter.writeLine(queryKey, sortedDocs);
        }
        resultWriter.closeWriter();
    }

    private static void evalAndCompare(String ... resultFiles) throws Exception {
        System.out.println("\nORIGNAL, WORD-EMBEDDING, ORIGINAL/WORD-EMBEDDING");
        for (String resultFile: resultFiles) {
             Parameters evalParams = setParams(new ArrayList<Pair>(Arrays.asList(Pair.with("metrics", "map"),
                                                                                 Pair.with("judgments", queryJudgement),
                                                                                 Pair.with("baseline", resultFile))),
                                               new ArrayList<Pair>(Arrays.asList(Pair.with("details", true))),
                                               new ArrayList<Pair>());


            QuerySetJudgments qsj = new QuerySetJudgments(queryJudgement);
            Parameters resultParams = Eval.singleEvaluation(evalParams, qsj, new ArrayList<>());

            Double map = ((Double) (resultParams.getMap("all").get("map")));
            System.out.println(String.format("MAP = %s, NDCG@10 = %s ", map.toString(), "X"));
        }
    }

    private static void waitOn(WordEmbeddedScorer t1, OtherModelScorer t2) {
        while(t1.isAlive() || t2.isAlive());
    }


    public static void main(String[] args) throws Exception {
        //write_d_hat();
        //write_q();

        Word2Vec queryWordVectors = null;//WordVectorSerializer.readWord2VecModel(new File(queryWordVectorPath));
        Word2Vec documentWordVectors = null;//WordVectorSerializer.readWord2VecModel(new File(dHatVectorPath));
        double alpha = 2;
        String otherModel = "bm25";
//        String otherModel = "jm";

        // bm25, Krovetz stemming, Dirchlet smoothing (mu=1000) ---------------------
        Parameters queryParams = setParams(new ArrayList<Pair>(Arrays.asList(Pair.with("queryFormat", "tsv"),
                                                                             Pair.with("query", queryPath),
                                                                             Pair.with("defaultTextPart", "postings.krovetz"),
                                                                             Pair.with("index", indexPath),
                                                                             Pair.with("scorer", otherModel))),
                                           new ArrayList<Pair>(),
                                           new ArrayList<Pair>(Arrays.asList(Pair.with("requested", 1000),
                                                                             Pair.with("mu", 1000)
//                                                                             Pair.with("lambda", 1)
                                                                             )));
        List<Parameters> queries = new BatchSearch().readParameters(queryPath);

        queryWordVectors = WordVectorSerializer.readWord2VecModel(new File(queryWordVectorPath));
        documentWordVectors = WordVectorSerializer.readWord2VecModel(new File(dHatVectorPath));

        for (Parameters q :  queries) {
            OtherModelScorer otherModelScorer = new OtherModelScorer(queryParams, q.getAsString("text"));
            WordEmbeddedScorer wordEmbeddedScorer = new WordEmbeddedScorer(documentWordVectors,
                                                                           queryWordVectors,
                                                                           getAllDocumentNames(),
                                                                           q.getAsString("number"));
            waitOn(otherModelScorer, wordEmbeddedScorer);
        }
//        HashMap<String, HashMap<String, Double>> otherModelResults = bs.retrieve(resultsPath, queryParams, otherModel);

//        queryWordVectors = WordVectorSerializer.readWord2VecModel(new File(queryWordVectorPath));
//        documentWordVectors = WordVectorSerializer.readWord2VecModel(new File(dHatVectorPath));
//        WordEmbeddedScorer wordEmbeddedScorer = new WordEmbeddedScorer(documentWordVectors, queryWordVectors, getAllDocumentNames());

//        HashMap<String, HashMap<String, Double>> wordEmbeddingResults = new QueryResultsIO().readFile(altResultsPath);

        HashMap<String, Double> wordEmbeddingResults = wordEmbeddedScorer.getFinalResults();
//        writeResults(wordEmbeddingResults, WEResultsPath);
//        normalize(documentResults, wordEmbeddingResults);
//        HashMap<String, HashMap<String, Double>> combined = combine(documentResults, wordEmbeddingResults, alpha);
//        writeResults(combined, combinedResultsPath);
//        evalAndCompare(resultsPath, WEResultsPath, combinedResultsPath);
    }
}
