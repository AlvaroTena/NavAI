import math
import sys


def print_usage():
    print(
        """
********************************************************************************
 This script should receive as arguments a KIPOS file and the output filepath
 Example Usage:
     python script.py INPUT_KIPOS_FILE OUTPUT_KML_FILE [Mode]
     Mode: 0 = Terrestrial (default)
           1 = Aeronautical
********************************************************************************
"""
    )


def parse_color_scale():
    return [
        {"color": "ff00ff00", "low": 0, "high": 1, "point": "P1"},
        {"color": "ff00ffff", "low": 1, "high": 3, "point": "P2"},
        {"color": "ff2400df", "low": 3, "high": 6, "point": "P3"},
    ]


def decimal_from_dms(degrees, minutes, seconds):
    sign = 1 if degrees >= 0 else -1
    return degrees + sign * minutes / 60 + sign * seconds / 3600


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nError in the input Arguments!!\n")
        print_usage()
        sys.exit(-1)

    kipos_file = sys.argv[1]
    kml_file = sys.argv[2]
    aeronautical = (
        int(sys.argv[3]) if len(sys.argv) == 4 and sys.argv[3] in ["0", "1"] else 0
    )

    try:
        with open(kipos_file, "r") as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"Can't open {kipos_file}")
        sys.exit(-1)

    try:
        outfile = open(kml_file, "w")
    except Exception as e:
        print(f"Can't open {kml_file}: {e}")
        sys.exit(-1)

    kipos_header = [line for line in lines if line.startswith("#")][0]
    station = kipos_header.split()[1]

    outfile.write(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Style id="P0">
      <IconStyle>
        <color>ffffffff</color>
        <scale>0.4</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/pal2/icon18.png</href></Icon>
      </IconStyle>
    </Style>
"""
    )

    color_scale = parse_color_scale()

    for scale in color_scale:
        outfile.write(
            f"""<Style id="{scale['point']}">
      <IconStyle>
        <color>{scale['color']}</color>
        <scale>0.2</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/pal2/icon18.png</href></Icon>
      </IconStyle>
    </Style>
"""
        )

    outfile.write(
        """<Style id="trackLine">
 <LineStyle>
 <color>ff007D00</color>
 <width>3</width>
 </LineStyle>
 </Style>
<Style id="headingLine">
 <LineStyle>
 <color>ffff0000</color>
 <width>2</width>
 </LineStyle>
 </Style>
<Folder>
  <name>Positions with timestamp</name>
"""
    )

    all_coords = ""
    headings = ""
    start_pos = end_pos = None

    epoch_count = 0

    for line in lines:
        if line.startswith("#"):
            continue

        epoch_count += 1
        if epoch_count >= 10:
            epoch_count = 0

        if epoch_count == 0:
            data = line.split()
            if len(data) < 50:
                continue

            date = f"<when>{data[0]}-{data[1].zfill(2)}-{data[2].zfill(2)}T{data[3].zfill(2)}:{data[4].zfill(2)}:{data[5].zfill(2)}Z</when>"

            lat = decimal_from_dms(float(data[9]), float(data[10]), float(data[11]))
            lon = decimal_from_dms(float(data[12]), float(data[13]), float(data[14]))
            height = float(data[15])

            sigma3d = float(data[34])
            point_color = "P1"
            for scale in color_scale:
                if scale["low"] <= sigma3d <= scale["high"]:
                    point_color = scale["point"]
                    break

            speed = (
                math.sqrt(
                    float(data[35]) ** 2 + float(data[36]) ** 2 + float(data[37]) ** 2
                )
                * 3.6
            )

            heading = (
                float(data[49]) / 57.2957795131
            )  # Convert heading from degrees to radians
            radxyz = math.sqrt(
                float(data[6]) ** 2 + float(data[7]) ** 2 + float(data[8]) ** 2
            )
            radxy = math.sqrt(float(data[6]) ** 2 + float(data[7]) ** 2)
            delta_lat = 5 * math.cos(heading) / radxyz * 57.2957795131
            delta_lon = 5 * math.sin(heading) / radxy * 57.2957795131

            outfile.write(f"    <Placemark>\n")
            outfile.write(f"      <TimeStamp>\n        {date}\n      </TimeStamp>\n")
            outfile.write(f"      <styleUrl>#{point_color}</styleUrl>\n")
            outfile.write(f"      <Point>\n")
            if aeronautical:
                outfile.write(
                    f"        <altitudeMode>absolute</altitudeMode>\n        <extrude>1</extrude>\n"
                )
            outfile.write(
                f"        <coordinates>{lon},{lat},{height}</coordinates>\n      </Point>\n"
            )

            infobox = f"""<description><![CDATA[<B>Epoch: </B><BR><BR>
                <TABLE border="1" width="100%" Align="center">
                    <TR ALIGN=RIGHT>
                    <TD ALIGN=LEFT>Time:</TD><TD>{data[0]}/{data[1]}/{data[2]}</TD><TD>{data[3]}:{data[4]}:{data[5]}</TD><TD></TD><TD></TD></TR>
                    <TR ALIGN=RIGHT><TD ALIGN=LEFT>Position:</TD><TD>{lat:.10f}</TD><TD>{lon:.10f}</TD><TD>{height:.4f}</TD><TD>(deg,m)</TD></TR>
                    <TR ALIGN=RIGHT><TD ALIGN=LEFT>Quality:</TD><TD>{int(sigma3d)}</TD><TD></TD><TD></TD><TD></TD></TR>
                    <TR ALIGN=RIGHT><TD ALIGN=LEFT>nSats(tot,gps,gal,bds):</TD><TD>{data[25]}</TD><TD>{data[26]}</TD><TD>{data[27]}</TD><TD>{data[28]}</TD></TR>
                    <TR ALIGN=RIGHT><TD ALIGN=LEFT>Vel:</TD><TD>{speed:.3f}</TD><TD>(km/h)</TD><TD></TD><TD></TD></TR>
                    <TR ALIGN=RIGHT><TD ALIGN=LEFT>Att(r,p,h):</TD><TD>{data[47]}</TD><TD>{data[48]}</TD><TD>{data[49]}</TD><TD>(deg)</TD></TR>
                </TABLE>
            ]]></description>"""

            outfile.write(infobox)
            outfile.write("    </Placemark>\n")

            headings += f"<Placemark><styleUrl>#headingLine</styleUrl><LineString><coordinates>{lon},{lat},{height} {lon + delta_lon},{lat + delta_lat},{height}</coordinates></LineString></Placemark>\n"

            if start_pos is None:
                start_pos = f"{lon},{lat},{height}"
            end_pos = f"{lon},{lat},{height}"
            all_coords += f"{lon},{lat},{height}\n"

    outfile.write(
        f"<Placemark>\n<name>Track</name>\n<Style><LineStyle><color>ff007D00</color><width>3</width></LineStyle></Style>\n<LineString>\n<coordinates>{all_coords}</coordinates>\n</LineString>\n</Placemark>\n"
    )
    outfile.write(f"<Folder><name>Headings</name>{headings}</Folder>\n")
    outfile.write("</Folder>\n")
    outfile.write("</Document>\n</kml>")
    outfile.close()


if __name__ == "__main__":
    main()
