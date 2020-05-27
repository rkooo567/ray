import {
  createStyles,
  TableCell,
  TableRow,
  Theme,
  withStyles,
  WithStyles,
} from "@material-ui/core";
import AddIcon from "@material-ui/icons/Add";
import RemoveIcon from "@material-ui/icons/Remove";
import classNames from "classnames";
import React, { useState } from "react";
import {
  MemoryTableResponse,
  MemoryTableEntry,
  MemoryTableSummary,
} from "../../../api";
import MemorySummary from "./MemorySummary";

const styles = (theme: Theme) =>
  createStyles({
    cell: {
      padding: theme.spacing(1),
      textAlign: "center",
      "&:last-child": {
        paddingRight: theme.spacing(1),
      },
    },
    expandCollapseCell: {
      cursor: "pointer",
    },
    expandCollapseIcon: {
      color: theme.palette.text.secondary,
      fontSize: "1.5em",
      verticalAlign: "middle",
    },
    extraInfo: {
      fontFamily: "SFMono-Regular,Consolas,Liberation Mono,Menlo,monospace",
      whiteSpace: "pre",
    },
  });

type Props = {
  groupKey: string;
  memoryTableGroups: MemoryTableResponse["group"];
  initialExpanded: boolean;
};

const MemoryRowGroup = (props: Props & WithStyles<typeof styles>) => {
  const { classes, groupKey, memoryTableGroups } = props;
  const [expanded, setExpanded] = useState(props.initialExpanded);
  const toggleExpanded = () => setExpanded(!expanded);

  const features = [
    "node_ip_address",
    "pid",
    "type",
    "object_id",
    "object_size",
    "reference_type",
    "call_site",
  ];

  const memoryTableGroup = memoryTableGroups[groupKey];
  const entries: Array<MemoryTableEntry> = memoryTableGroup["entries"];
  const summary: MemoryTableSummary = memoryTableGroup["summary"];

  return (
    <React.Fragment>
      <TableRow hover>
        <TableCell
          className={classNames(classes.cell, classes.expandCollapseCell)}
          onClick={toggleExpanded}
        >
          {!expanded ? (
            <AddIcon className={classes.expandCollapseIcon} />
          ) : (
            <RemoveIcon className={classes.expandCollapseIcon} />
          )}
        </TableCell>
        {features.map((feature, index) => (
          <TableCell className={classes.cell} key={index}>
            {// TODO(sang): For now, it is always grouped by node_ip_address.
            feature === "node_ip_address" ? groupKey : ""}
          </TableCell>
        ))}
      </TableRow>
      {expanded && (
        <React.Fragment>
          <MemorySummary initialExpanded={false} memoryTableSummary={summary} />
          {entries.map((memory_table_entry, index) => {
            const object_size =
              memory_table_entry.object_size === -1
                ? "?"
                : memory_table_entry.object_size + " B";
            return (
              <TableRow hover key={index}>
                <TableCell className={classes.cell} />
                <TableCell className={classes.cell}>
                  {memory_table_entry.node_ip_address}
                </TableCell>
                <TableCell className={classes.cell}>
                  {memory_table_entry.pid}
                </TableCell>
                <TableCell className={classes.cell}>
                  {memory_table_entry.type}
                </TableCell>
                <TableCell className={classes.cell}>
                  {memory_table_entry.object_id}
                </TableCell>
                <TableCell className={classes.cell}>{object_size}</TableCell>
                <TableCell className={classes.cell}>
                  {memory_table_entry.reference_type}
                </TableCell>
                <TableCell className={classes.cell}>
                  {memory_table_entry.call_site}
                </TableCell>
              </TableRow>
            );
          })}
        </React.Fragment>
      )}
    </React.Fragment>
  );
};

export default withStyles(styles)(MemoryRowGroup);
