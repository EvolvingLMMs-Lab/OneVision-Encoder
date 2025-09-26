package storm.petrel;

import java.io.InputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.util.Map;

import org.apache.storm.thrift.TBase;

import org.yaml.snakeyaml.Yaml;

import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.StormSubmitter;
import org.apache.storm.generated.StormTopology;

public class GenericTopology
{
    private static StormTopology readTopology() throws IOException
    {
        InputStream stream = ResourceLoader.getResourceAsStream("resources/topology.ser", GenericTopology.class);
        assert stream != null;
        
        try
        {
            // Create the reader
            ThriftReader thriftIn = new ThriftReader(stream, new ThriftReader.TBaseCreator() {
              @Override public TBase create() {
                return new StormTopology();
              }
            });
            
            // Read objects
            assert thriftIn.hasNext();
            TBase base = thriftIn.read();
            assert base != null;
            StormTopology topology = (StormTopology) base;
            assert topology != null;
            
            return topology;
        }
        finally
        {
            // Close stream
            stream.close();
        }
    }

    public static void main(String[] args) throws Exception
    {
        // Read topology file created by petrelbuilder.
        StormTopology topology = readTopology();

        // Get topology-specific configuration from the JAR.
        Config conf = new Config();
        InputStream stream = ResourceLoader.getResourceAsStream("resources/__topology__.yaml", GenericTopology.class);
        try
        {
            Yaml yaml = new Yaml();
            Map localConf = (Map) yaml.load(new InputStreamReader(stream));
            conf.putAll(localConf);
        }
        finally
        {
            // Close stream
            stream.close();
        }

        // Read __submitter__.yaml to get submitter user and hostname
        // information. Include it in the config.
        stream = ResourceLoader.getResourceAsStream("resources/__submitter__.yaml", GenericTopology.class);
        try
        {
            Yaml yaml = new Yaml();
            Map submitterConf = (Map) yaml.load(new InputStreamReader(stream));
            conf.putAll(submitterConf);
        }
        finally
        {
            // Close stream
            stream.close();
        }
        
        if (args!=null && args.length > 0)
        {
            StormSubmitter.submitTopology(args[0], conf, topology);
        }
        else
        {
            // Force some conservative settings to try and avoid overloading a
            // local machine's CPU or memory.
            conf.setDebug(true);
            conf.setNumWorkers(1);
            conf.setMaxTaskParallelism(1);

            final LocalCluster cluster = new LocalCluster();
            cluster.submitTopology("test topology", conf, topology);

            Runtime.getRuntime().addShutdownHook(new Thread() {
                public void run() {
                    System.out.println("Shutting down local topology");
                    cluster.shutdown();
                }
            });    

            while (true)
            {
                System.out.println("Topology is running. Press ^C to stop it.");
                Thread.sleep(60000);
            }
        }
    }
}
