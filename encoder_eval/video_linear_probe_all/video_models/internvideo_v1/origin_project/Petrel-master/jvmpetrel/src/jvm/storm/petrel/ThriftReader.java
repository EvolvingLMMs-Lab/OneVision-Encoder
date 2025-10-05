package storm.petrel;

// Adapted from http://joelpm.com/2009/02/05/thrift-reading-thrift-objects-from-disk-with-java.html
// Original code reads from a file. Changed it to read from a stream.
import java.io.InputStream;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.storm.thrift.TBase;
import org.apache.storm.thrift.TException;
import org.apache.storm.thrift.protocol.TBinaryProtocol;
import org.apache.storm.thrift.transport.TIOStreamTransport;

/**
 * A simple class for reading Thrift objects (of a single type) from a file.
 *
 * @author Joel Meyer
 */
public class ThriftReader {
  /**
    * Thrift deserializes by taking an existing object and populating it. ThriftReader
    * needs a way of obtaining instances of the class to be populated and this interface
    * defines the mechanism by which a client provides these instances.
    */
  public static interface TBaseCreator {
    TBase create();
  }

  /** Used to create empty objects that will be initialized with values from the file. */
  protected final TBaseCreator creator;
  
  /** For reading the file. */
  // private BufferedInputStream bufferedIn;
  private InputStream inputStream;

  /** For reading the binary thrift objects. */
  private TBinaryProtocol binaryIn;
  
  /**
    * Constructor.
    */
  public ThriftReader(InputStream stream, TBaseCreator creator) {
    this.inputStream = stream;
    this.creator = creator;
    binaryIn = new TBinaryProtocol(new TIOStreamTransport(inputStream));
  }
  
  /**
    * Opens the file for reading. Must be called before {@link read()}.
    */
  public void open() throws FileNotFoundException {
    //inputStream = new BufferedInputStream(new FileInputStream(file), 2048);
    //binaryIn = new TBinaryProtocol(new TIOStreamTransport(inputStream));
  }

  /**
    * Checks if another objects is available by attempting to read another byte from the stream.
    */
  public boolean hasNext() throws IOException {
    inputStream.mark(1);
    int val = inputStream.read();
    inputStream.reset();
    return val != -1;
  }

  /**
    * Reads the next object from the file.
    */
  public TBase read() throws IOException {
    TBase t = creator.create();
    try {
      t.read(binaryIn);
    } catch (TException e) {
      throw new IOException(e);
    }
    return t;
  }
}
