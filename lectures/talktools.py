"""Tools to style a talk."""

from IPython.display import HTML, display, YouTubeVideo

def prefix(url):
    prefix = '' if url.startswith('http') else 'http://'
    return prefix + url


def simple_link(url, name=None):
    name = url if name is None else name
    url = prefix(url)
    return '<a href="%s" target="_blank">%s</a>' % (url, name)


def html_link(url, name=None):
    return HTML(simple_link(url, name))


# Utility functions
def website(url, name=None, width=800, height=450):
    html = []
    if name:
        html.extend(['<div class="nb_link">',
                     simple_link(url, name),
                     '</div>'] )

    html.append('<iframe src="%s"  width="%s" height="%s">' % 
                (prefix(url), width, height))
    return HTML('\n'.join(html))


def nbviewer(url, name=None, width=800, height=450):
    return website('nbviewer.ipython.org/url/' + url, name, width, height)


# Load and publish CSS
style = HTML("""
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>

<style>

@import url(https://fonts.googleapis.com/css?family=Open+Sans);body{
   font-family: 'Open Sans';
   font-size: 125%;
}

.talk_title
{
  color: #498AF3;
  font-size: 275%;
  font-weight:bold;
  line-height: 1.3; 
  margin: 10px 50px 10px;
  }

.subtitle
{
  color: #386BBC;
  font-size: 180%;
  font-weight:bold;
  line-height: 1.2; 
  margin: 20px 50px 20px;
  }

.rendered_html h1
{
  color: #498AF3;
  line-height: 1.2; 
  margin: 0.15em 0em 0.5em;
  page-break-before: always;
  text-align: center;
  }


.center
{
  text-align: center;
  }

.nb_link
{
    padding-bottom: 0.5em;
}

</style>""")


display(style)
