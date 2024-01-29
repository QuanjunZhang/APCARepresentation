import re

"""
.dot文件解析
"""


def parse_dot_to_graph(dot_path: str, output_path: str = None, is_joern=False):
    if output_path is None:
        output_path = dot_path
    edges = []
    nodes = []
    with open(dot_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            if is_joern:
                line = line.replace("&lt;", "<").replace("&gt;", ">")
            if is_start(line):
                pass
            elif is_edge(line):
                if not is_joern:
                    edges.append(parse_dot_edge(line))
                else:
                    edges.append(parse_dot_edge_for_joern(line))
            elif is_joern and is_node_for_joern(line):
                nodes.append(parse_dot_node_for_joern(line))
            elif not is_joern and is_node(line):
                nodes.append(parse_dot_node(line))
            elif is_end(line):
                break
    # remove redundant nodes
    nodes = list(
        filter(lambda node: node['id'] in [x['start'] for x in edges] or node['id'] in [x['end'] for x in edges],
               nodes))

    # output
    index = output_path.rfind('.')

    if index >= 0:
        output_path = output_path[:index]

    edge_output = output_path + ".edge"
    node_output = output_path + ".node"
    write_edges(edges, edge_output)
    write_nodes(nodes, node_output)


def is_start(line: str):
    return line.startswith("digraph") or line.startswith("subgraph") or line.startswith("label")


def is_end(line: str):
    return line.strip() == "}"


def is_edge(line: str):
    return "->" in line


def is_node(line: str):
    return "style =" in line and "label = " in line and "->" not in line


# e.g. "32" [label = <(IDENTIFIER,MYSQL_DB_USERNAME,setUser(MYSQL_DB_USERNAME))<SUB>5</SUB>> ]
def is_node_for_joern(line: str):
    return re.match("\"[0-9]+\" \\[label = <.*<SUB>[0-9]+</SUB>> ]", line)


def parse_dot_edge(line: str):
    mm = re.match("0\\.([0-9]+) -> 0\\.([0-9]+)", line.strip())
    return {
        "start": mm.group(1),
        "end": mm.group(2),
    }


# e.g.   "41" -> "45"  [ label = "&lt;RET&gt;"]
def parse_dot_edge_for_joern(line: str):
    mm = re.search("\"([0-9]+)\" -> \"([0-9]+)\"", line.strip())
    return {
        "start": mm.group(1),
        "end": mm.group(2),
    }


def parse_dot_node(line: str):
    mm = re.match("0\\.([0-9]+) \\[style = filled, label = \"(.*) <.*>\", type = (.*), fillcolor =", line)
    mmm = re.match("0\\.([0-9]+)", line)
    if mm is None:
        return {
            "id": mmm.group(1),
            "type": 'Block',
            "code": "{}",
        }
    return {
        "id": mm.group(1),
        "type": mm.group(3).strip(),
        "code": mm.group(2).strip(),
    }


def parse_dot_node_for_joern(line: str):
    mm = re.match("\"([0-9]+)\" \\[label = <\\(([^,]+),(.*)\\)<SUB>[0-9]+</SUB>> ]",
                  line)
    return {
        "id": mm.group(1),
        "type": mm.group(2).strip(),
        "code": mm.group(3).strip(),
    }


def write_edges(edges, f):
    s = ""
    for e in edges:
        if s != "":
            s += "\n"
        s += "{} -> {}".format(e['start'], e['end'])

    with open(f, 'w', encoding="utf8") as ff:
        ff.write(s)
        ff.flush()
        ff.close()


def write_nodes(nodes, f):
    s = ""
    for n in nodes:
        if s != "":
            s += "\n"
        s += "id:{} type:{} code:{}".format(n['id'], n['type'], n['code'])
    with open(f, 'w', encoding="utf8") as ff:
        ff.write(s)
        ff.flush()
        ff.close()
