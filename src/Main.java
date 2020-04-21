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
import org.apache.xalan.lib.sql.QueryParameter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.lemurproject.galago.core.eval.Eval;
import java.lang.Math;
import org.lemurproject.galago.core.eval.QuerySetJudgments;
import org.lemurproject.galago.core.index.disk.DiskIndex;
import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;
//import org.lemurproject.galago.core.tools.apps.BatchSearch;
import org.javatuples.Pair;

import java.io.*;;

import java.util.*;

import static org.lemurproject.galago.core.tools.apps.BatchSearch.logger;

/**
 * Kyle Price
 * Brandon Mouser
 */
public class Main {
    public static final String indexPath = "index";
    public static final String queryPath = "PsgRobust/robust04.titles.tsv";
    public static final String queryJudgement = "PsgRobust/robust04.qrels";
    public static final String wordvecPath = "GoogleNews-vectors-negative300.bin";
    public static final String queryWordVectorPath = "QUERY_WORDS.txt";
    public static final String dHatVectorPath = "D_HAT.txt";
    public static final int requested = 1000;

    public static final String batchResultsPath = "batchResults";
    public static final String resultsPathTfIdf = "combinedResultsTfIdf";
    public static final String resultsPathBm25 = "combinedResultsBm25";
    public static final String standaloneBm25 = "resultsBm25";
    public static final String standaloneTfIdf = "resultsTfIfd";
    public static final String resultsPathWe = "resultsWe";

    public static String resultsPath;
    public static String standalonePath;
    public static String scorer;
    public static String opWrapper;


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
            if(index.containsDocumentIdentifier((long) i)) {
                documentNames.add(index.getName((long) i));
            }
        }
        return documentNames;
    }

    private static void evalAndCompare(String ... resultFiles) throws Exception {
        for (String resultFile: resultFiles) {
             Parameters evalParams = setParams(new ArrayList<>(Arrays.asList(Pair.with("judgments", queryJudgement),
                                                                             Pair.with("baseline", resultFile))),
                                               new ArrayList<>(Arrays.asList(Pair.with("details", true))),
                                               new ArrayList<>());
            evalParams.set("metrics", Arrays.asList("ndcg10", "map"));


            QuerySetJudgments qsj = new QuerySetJudgments(queryJudgement);
            Parameters resultParams = Eval.singleEvaluation(evalParams, qsj, new ArrayList<>());

            Double map = ((Double) (resultParams.getMap("all").get("map")));
            Double ndcg = ((Double) (resultParams.getMap("all").get("ndcg10")));
            System.out.println(String.format("%s: MAP = %s, NDCG@10 = %s ", resultFile, map.toString(), ndcg.toString()));
        }
    }

    private static void waitOn(Thread ... threads) {
        try {
            for (Thread t : threads)
                t.join();
        }
        catch(java.lang.InterruptedException e) {
            System.out.println("an error occured while joining threads");
        }
    }

    public static HashMap<String, Double> execQuery(Parameters query, Retrieval ret, int numDocs) {
        HashMap<String, Double> resultDocuments = new HashMap<>();
        try {
            String queryNumber = query.getString("number");
            String queryText = query.getString("text");
            Node root = StructuredQuery.parse(queryText);
            Node transformed = ret.transformQuery(root, query);
            query.set("requested", numDocs);

            // run query
            List<ScoredDocument> results = ret.executeQuery(transformed, query).scoredDocuments;
            QueryResultsIO.resultDocuments(queryNumber, results, resultDocuments);

        } catch (java.lang.Exception e) {
            System.out.println("Error in OtherModelScorer Thread!!!");
        }
        return resultDocuments;
    }

    public static void runBatchSearch(String outputFileName, Parameters queryParams, List<Parameters> queries) throws Exception {

        Retrieval retrieval = RetrievalFactory.create(queryParams);
        QueryResultsIO writer = new QueryResultsIO();
        writer.newWriter(outputFileName);

        // for each query, run it, get the results, print in TREC format
        for (Parameters query : queries) {
            String queryNumber = query.getString("number");
            String queryText = query.getString("text");
            queryText = queryText.toLowerCase(); // option to fold query cases -- note that some parameters may require upper case

            query.set("requested", requested);

            Node root = StructuredQuery.parse(queryText);
            Node transformed = retrieval.transformQuery(root, query);

            // run query
            List<ScoredDocument> results = retrieval.executeQuery(transformed, query).scoredDocuments;
            writer.writeLines(queryNumber, results);
        }
        writer.closeWriter();
    }

    public static Parameters chooseModel(String model) throws Exception {
        List<String> documents = getAllDocumentNames();
        Parameters queryParams;
        switch (model) {
            case "tf_idf":
                resultsPath = resultsPathTfIdf;
                standalonePath = standaloneTfIdf;
                scorer = "jm";
                opWrapper = "";
                queryParams =  setParams(new ArrayList<>(Arrays.asList(Pair.with("queryFormat", "tsv"),
                                                                       Pair.with("query", queryPath),
                                                                       Pair.with("defaultTextPart", "postings.krovetz"),
                                                                       Pair.with("index", indexPath),
                                                                       Pair.with("scorer", scorer))),
                                         new ArrayList<>(),
                                         new ArrayList<>(Arrays.asList(Pair.with("requested", documents.size()))));
                break;
            case "bm25":
                resultsPath = resultsPathBm25;
                standalonePath = standaloneBm25;
                opWrapper = scorer = "bm25";
                queryParams = setParams(new ArrayList<>(Arrays.asList(Pair.with("queryFormat", "tsv"),
                                                                      Pair.with("query", queryPath),
                                                                      Pair.with("defaultTextPart", "postings.krovetz"),
                                                                      Pair.with("index", indexPath),
                                                                      Pair.with("scorer", scorer))),
                                       new ArrayList<>(),
                                       new ArrayList<>(Arrays.asList(Pair.with("requested", documents.size()))));
                break;
            case "we":
                resultsPath = resultsPathWe;

                // batch results don't matter cuz we're not using the 'other' model anyway
                standalonePath = standaloneTfIdf;
                scorer = "jm";
                opWrapper = "";
                queryParams = setParams(new ArrayList<>(Arrays.asList(Pair.with("queryFormat", "tsv"),
                                                                      Pair.with("query", queryPath),
                                                                      Pair.with("defaultTextPart", "postings.krovetz"),
                                                                      Pair.with("index", indexPath),
                                                                      Pair.with("scorer", scorer))),
                                       new ArrayList<>(),
                                       new ArrayList<>(Arrays.asList(Pair.with("requested", documents.size()))));
                break;
            default:
                throw new Exception("invalid model selection");
        }
        queryParams.set("requested", documents.size());
        return queryParams;
    }



    public static void main(String[] args) throws Exception {
//        write_d_hat();
//        write_q();

        double alpha = 0.5;
        Parameters queryParams = chooseModel("bm25");

        List<Parameters> queries = QueryResultsIO.readParameters(queryPath, opWrapper);
        runBatchSearch(standalonePath, queryParams, queries);
//        evalAndCompare(standaloneTfIdf, resultsPathTfIdf, standaloneBm25, resultsPathBm25);

//        System.exit(0);

        Word2Vec queryWordVectors = WordVectorSerializer.readWord2VecModel(new File(queryWordVectorPath));
        Word2Vec documentWordVectors = WordVectorSerializer.readWord2VecModel(new File(dHatVectorPath));
        System.out.println("Done reading vector files");

        Retrieval ret = RetrievalFactory.create(queryParams);
        QueryResultsIO writer = new QueryResultsIO();
        writer.newWriter(resultsPath);

        List<String> documents = getAllDocumentNames();
        Iterator it = queries.iterator();
        while (it.hasNext()) {
            // time it
            long start = System.nanoTime();

            Parameters q1 = (Parameters)it.next();
            HashMap<String, Double> embeddedResults1 = new HashMap<>();
            HashMap<String, Double> embeddedResults2 = new HashMap<>();
            HashMap<String, Double> embeddedResults3 = new HashMap<>();
            HashMap<String, Double> embeddedResults4 = new HashMap<>();

            HashMap<String, Double> otherResults1 = execQuery(q1, ret, documents.size());
            WordEmbeddedScorer wordEmbeddedScorer1 = new WordEmbeddedScorer(embeddedResults1,
                                                                           documentWordVectors,
                                                                           queryWordVectors,
                                                                           documents,
                                                                           QueryResultsIO.createQueryPair(queryPath),
                                                                           q1.getAsString("number"),
                                                                           otherResults1);
            if (!it.hasNext()) {
                waitOn(wordEmbeddedScorer1);
                ModelMixer.mix( otherResults1, embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
                break;
            }
            Parameters q2 = (Parameters)it.next();
            HashMap<String, Double> otherResults2 = execQuery(q2, ret, documents.size());
            WordEmbeddedScorer wordEmbeddedScorer2 = new WordEmbeddedScorer(embeddedResults2,
                                                                           documentWordVectors,
                                                                           queryWordVectors,
                                                                           documents,
                                                                           QueryResultsIO.createQueryPair(queryPath),
                                                                           q2.getAsString("number"),
                                                                           otherResults2);
            if (!it.hasNext()) {
                waitOn(wordEmbeddedScorer1, wordEmbeddedScorer2);
                ModelMixer.mix( otherResults1, embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
                ModelMixer.mix( otherResults2, embeddedResults2, writer, q2.getAsString("number"), alpha, requested);
                break;
            }
            Parameters q3 = (Parameters)it.next();
            HashMap<String, Double> otherResults3 = execQuery(q3, ret, documents.size());
            WordEmbeddedScorer wordEmbeddedScorer3 = new WordEmbeddedScorer(embeddedResults3,
                                                                           documentWordVectors,
                                                                           queryWordVectors,
                                                                           documents,
                                                                           QueryResultsIO.createQueryPair(queryPath),
                                                                           q3.getAsString("number"),
                                                                           otherResults3);

            if (!it.hasNext()) {
                waitOn(wordEmbeddedScorer1, wordEmbeddedScorer2, wordEmbeddedScorer3);
                ModelMixer.mix( otherResults1, embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
                ModelMixer.mix( otherResults2, embeddedResults2, writer, q2.getAsString("number"), alpha, requested);
                ModelMixer.mix( otherResults3, embeddedResults3, writer, q3.getAsString("number"), alpha, requested);
                break;
            }
            Parameters q4 = (Parameters)it.next();
            HashMap<String, Double> otherResults4 = execQuery(q4, ret, documents.size());
            WordEmbeddedScorer wordEmbeddedScorer4 = new WordEmbeddedScorer(embeddedResults4,
                                                                            documentWordVectors,
                                                                            queryWordVectors,
                                                                            documents,
                                                                            QueryResultsIO.createQueryPair(queryPath),
                                                                            q4.getAsString("number"),
                                                                            otherResults4);

            // wait on all
            waitOn(wordEmbeddedScorer1, wordEmbeddedScorer2, wordEmbeddedScorer3, wordEmbeddedScorer4);

            // linearly combine
            ModelMixer.mix(otherResults1, embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
            ModelMixer.mix(otherResults2, embeddedResults2, writer, q2.getAsString("number"), alpha, requested);
            ModelMixer.mix(otherResults3, embeddedResults3, writer, q3.getAsString("number"), alpha, requested);
            ModelMixer.mix(otherResults4, embeddedResults4, writer, q4.getAsString("number"), alpha, requested);

            long end = System.nanoTime();
            long duration = (end - start);

            System.out.print(String.format("\rProcessed queries: %s, %s, %s %s in %.0f seconds",
                                           q1.getAsString("number"),
                                           q2.getAsString("number"),
                                           q3.getAsString("number"),
                                           q4.getAsString("number"),
                                           duration / Math.pow(10, 9)));
        }
        System.out.println();
        evalAndCompare(standaloneTfIdf, resultsPathTfIdf, standaloneBm25, resultsPathBm25, resultsPathWe);
    }
}
