package storm.petrel;

// From http://www.java2s.com/Code/Java/File-Input-Output/Loadaresourceasastream.htm
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

public class ResourceLoader {

  /**
   * Load a given resource. <p/> This method will try to load the resource
   * using the following methods (in order):
   * <ul>
   * <li>From Thread.currentThread().getContextClassLoader()
   * <li>From ClassLoaderUtil.class.getClassLoader()
   * <li>callingClass.getClassLoader()
   * </ul>
   * 
   * @param resourceName The name of the resource to load
   * @param callingClass The Class object of the calling object
   */
  public static URL getResource(String resourceName, Class callingClass) {
      URL url = Thread.currentThread().getContextClassLoader().getResource(resourceName);
      if (url == null && resourceName.startsWith("/")) {
          //certain classloaders need it without the leading /
          url = Thread.currentThread().getContextClassLoader()
              .getResource(resourceName.substring(1));
      }

      ClassLoader cluClassloader = ResourceLoader.class.getClassLoader();
      if (cluClassloader == null) {
          cluClassloader = ClassLoader.getSystemClassLoader();
      }
      if (url == null) {
          url = cluClassloader.getResource(resourceName);
      }
      if (url == null && resourceName.startsWith("/")) {
          //certain classloaders need it without the leading /
          url = cluClassloader.getResource(resourceName.substring(1));
      }

      if (url == null) {
          ClassLoader cl = callingClass.getClassLoader();

          if (cl != null) {
              url = cl.getResource(resourceName);
          }
      }

      if (url == null) {
          url = callingClass.getResource(resourceName);
      }
      
      if ((url == null) && (resourceName != null) && (resourceName.charAt(0) != '/')) {
          return getResource('/' + resourceName, callingClass);
      }

      return url;
  }
  
  /**
   * This is a convenience method to load a resource as a stream. <p/> The
   * algorithm used to find the resource is given in getResource()
   * 
   * @param resourceName The name of the resource to load
   * @param callingClass The Class object of the calling object
   */
  public static InputStream getResourceAsStream(String resourceName, Class callingClass) {
      assert resourceName != null;
      assert callingClass != null;
      
      URL url = getResource(resourceName, callingClass);

      try {
          if (url != null)
          {
            InputStream result = url.openStream();
            assert result != null;
            return result;
          }
          else
          {
            return null;
          }
          //return (url != null) ? url.openStream() : null;
      } catch (IOException e) {
          System.err.println("IOException");
          System.err.println(e.getMessage());
          return null;
      }
  }

  /**
   * Load a given resources. <p/> This method will try to load the resources
   * using the following methods (in order):
   * <ul>
   * <li>From Thread.currentThread().getContextClassLoader()
   * <li>From ClassLoaderUtil.class.getClassLoader()
   * <li>callingClass.getClassLoader()
   * </ul>
   * 
   * @param resourceName The name of the resource to load
   * @param callingClass The Class object of the calling object
   */
  public static List<URL> getResources(String resourceName, Class callingClass) {
      List<URL> ret = new ArrayList<URL>();
      Enumeration<URL> urls = new Enumeration<URL>() {
          public boolean hasMoreElements() {
              return false;
          }
          public URL nextElement() {
              return null;
          }
          
      };
      try {
          urls = Thread.currentThread().getContextClassLoader()
              .getResources(resourceName);
      } catch (IOException e) {
          //ignore
      }
      if (!urls.hasMoreElements() && resourceName.startsWith("/")) {
          //certain classloaders need it without the leading /
          try {
              urls = Thread.currentThread().getContextClassLoader()
                  .getResources(resourceName.substring(1));
          } catch (IOException e) {
              // ignore
          }
      }

      ClassLoader cluClassloader = ResourceLoader.class.getClassLoader();
      if (cluClassloader == null) {
          cluClassloader = ClassLoader.getSystemClassLoader();
      }
      if (!urls.hasMoreElements()) {
          try {
              urls = cluClassloader.getResources(resourceName);
          } catch (IOException e) {
              // ignore
          }
      }
      if (!urls.hasMoreElements() && resourceName.startsWith("/")) {
          //certain classloaders need it without the leading /
          try {
              urls = cluClassloader.getResources(resourceName.substring(1));
          } catch (IOException e) {
              // ignore
          }
      }

      if (!urls.hasMoreElements()) {
          ClassLoader cl = callingClass.getClassLoader();

          if (cl != null) {
              try {
                  urls = cl.getResources(resourceName);
              } catch (IOException e) {
                  // ignore
              }
          }
      }

      if (!urls.hasMoreElements()) {
          URL url = callingClass.getResource(resourceName);
          if (url != null) {
              ret.add(url);
          }
      }
      while (urls.hasMoreElements()) {
          ret.add(urls.nextElement());
      }

      
      if (ret.isEmpty() && (resourceName != null) && (resourceName.charAt(0) != '/')) {
          return getResources('/' + resourceName, callingClass);
      }
      return ret;
  }



}